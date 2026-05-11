import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from flask import Blueprint, request, g
from pydantic import ValidationError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.middleware.auth import require_auth, require_role
from app.services.hr_service import HRService
from app.utils.response import success_response, error_response, validation_error_response
from app.utils.job_store import create_job, get_job, set_running, set_completed, set_failed
from app.schemas.cv_evaluation import CVEvaluationRequest, CVEvaluationResponse
from app.schemas.policy import PolicyUploadRequest, PolicyQuestionRequest, PolicyQuestionResponse
from app.schemas.technical import (
    TechnicalQuestionGenerateRequest, TechnicalQuestionResponse,
    TechnicalAnswerEvaluateRequest, TechnicalAnswerEvaluateResponse
)

bp = Blueprint('hr', __name__)
logger = logging.getLogger(__name__)
hr_service = HRService()

# When sync POST has more than this many CVs, we run async and return 202 + job_id
# to avoid long-running requests breaking uvicorn/asgiref (CurrentThreadExecutor).
SYNC_CV_FILE_LIMIT = 5


@bp.route('/cv/evaluate', methods=['POST'])
@require_auth
def evaluate_cvs():
    """
    Evaluate Multiple CVs
    Upload multiple candidate CVs and evaluate them against a job description
    ---
    tags:
      - HR AI Platform
    consumes:
      - multipart/form-data
    produces:
      - application/json
    security:
      - Bearer: []
    parameters:
      - in: formData
        name: job_description
        type: string
        required: true
        description: Job description text
      - in: formData
        name: cv_files
        type: array
        items:
          type: file
        collectionFormat: multi
        required: true
        description: Candidate CV files (PDF or DOCX, multiple allowed)
    responses:
      200:
        description: CV evaluation completed successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: CV evaluation completed
            data:
              type: object
              properties:
                results:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                        example: candidate1.pdf
                      score:
                        type: number
                        example: 85.5
                      evaluation:
                        type: string
                      skill_scores:
                        type: object
                      skill_status:
                        type: object
                      hire_recommendation:
                        type: object
                executive_kpis:
                  type: object
                  properties:
                    total_candidates:
                      type: integer
                    average_match:
                      type: number
                    top_score:
                      type: number
                    top_5_count:
                      type: integer
      400:
        description: Bad request (missing files or job description)
      401:
        description: Unauthorized
      500:
        description: Server error
    """
    try:
        # Get job description from form data
        jd_text = request.form.get('job_description')
        if not jd_text:
            return error_response("job_description is required", status_code=400)

        # Get CV files
        cv_files = request.files.getlist('cv_files')
        if not cv_files or not any(f.filename for f in cv_files):
            return error_response("At least one CV file is required", status_code=400)

        # Avoid long-running sync request under uvicorn/asgiref (CurrentThreadExecutor).
        # Use async flow and return 202 + job_id so client can poll.
        if len(cv_files) > SYNC_CV_FILE_LIMIT:
            temp_dir = tempfile.mkdtemp(prefix="cv_job_")
            file_names = []
            try:
                for i, f in enumerate(cv_files):
                    name = secure_filename(f.filename) or f"file_{i}"
                    if not name:
                        name = f"file_{i}"
                    path = os.path.join(temp_dir, name)
                    f.save(path)
                    file_names.append(name)
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return error_response(f"Failed to save uploads: {str(e)}", status_code=500)
            job_id = create_job()
            thread = threading.Thread(
                target=_run_cv_evaluation_async,
                args=(job_id, jd_text, temp_dir, file_names),
                daemon=True,
            )
            thread.start()
            return success_response(
                data={"job_id": job_id, "status": "pending"},
                message="Large batch: poll GET /api/hr/cv/evaluate/status/<job_id> for result",
                status_code=202,
            )

        # Evaluate CVs (sync for small batches)
        result = hr_service.evaluate_cvs(cv_files, jd_text)

        # Format response
        response_data = CVEvaluationResponse(
            results=result["results"],
            executive_kpis=result["executive_kpis"],
            processing_time_seconds=result.get("processing_time_seconds"),
        )

        return success_response(data=response_data.dict(), message="CV evaluation completed")

    except Exception as e:
        return error_response(f"Error evaluating CVs: {str(e)}", status_code=500)


def _cv_worker_send_result(result_queue: multiprocessing.SimpleQueue, payload: tuple) -> None:
    """
    Put the job result on the shared queue.

    Uses SimpleQueue (pipe-backed, synchronous put) so the message is fully
    handed off before the worker exits. The standard multiprocessing.Queue uses
    a background feeder thread; the process can exit with code 0 before bytes
    reach the parent, so the collector never sees the job.
    """
    result_queue.put(payload)


def _cv_evaluation_worker_process(
    job_id: str, jd_text: str, temp_dir: str, file_names: list, result_queue: multiprocessing.SimpleQueue
) -> None:
    """
    Runs in a separate process so uvicorn/asgiref CurrentThreadExecutor is not broken.
    Puts (job_id, 'completed'|'failed', result_dict|error_str) on result_queue.
    """
    file_handles = []
    try:
        from app.schemas.cv_evaluation import CVEvaluationResponse
        from app.services.hr_service import HRService
        from werkzeug.datastructures import FileStorage

        logging.basicConfig(level=logging.INFO)
        wlog = logging.getLogger("app.api.hr.cv_worker")
        wlog.info("CV worker started job_id=%s files=%s", job_id, file_names)

        hr = HRService()
        cv_files = []
        for filename in file_names:
            path = os.path.join(temp_dir, filename)
            if not os.path.isfile(path):
                continue
            f = open(path, "rb")
            file_handles.append(f)
            cv_files.append(
                FileStorage(stream=f, filename=filename, content_type="application/octet-stream")
            )
        if not cv_files:
            _cv_worker_send_result(result_queue, (job_id, "failed", "No valid CV files found"))
            return
        result = hr.evaluate_cvs(cv_files, jd_text)
        response_data = CVEvaluationResponse(
            results=result["results"],
            executive_kpis=result["executive_kpis"],
            processing_time_seconds=result.get("processing_time_seconds"),
        )
        wlog.info("CV worker finished job_id=%s; sending result", job_id)
        _cv_worker_send_result(result_queue, (job_id, "completed", response_data.dict()))
    except Exception as e:
        logging.getLogger("app.api.hr.cv_worker").exception(
            "CV worker failed job_id=%s", job_id
        )
        _cv_worker_send_result(result_queue, (job_id, "failed", str(e)))
    finally:
        for f in file_handles:
            try:
                f.close()
            except Exception:
                pass
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# SimpleQueue = pipe-backed; avoids multiprocessing.Queue feeder-thread drops on worker exit.
_cv_result_queue: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()


def _cv_result_collector() -> None:
    """Background thread: read from _cv_result_queue and update job_store."""
    while True:
        try:
            msg = _cv_result_queue.get()
            if msg is None:
                break
            job_id, status, data = msg[0], msg[1], msg[2]
            if status == "completed":
                try:
                    set_completed(job_id, data)
                    logger.info("CV result collector: job_id=%s completed", job_id)
                except Exception:
                    logger.exception(
                        "CV result collector: set_completed failed job_id=%s", job_id
                    )
                    set_failed(job_id, "Internal error storing evaluation result")
            else:
                set_failed(job_id, data)
                logger.info("CV result collector: job_id=%s failed", job_id)
        except Exception as e:
            logger.exception("CV result collector error: %s", e)


def _start_cv_result_collector_if_main() -> None:
    """Start the result collector thread only in the main process (not in worker subprocesses)."""
    if multiprocessing.current_process().name == "MainProcess":
        _t = threading.Thread(target=_cv_result_collector, daemon=True)
        _t.start()


_start_cv_result_collector_if_main()


def _run_cv_evaluation_async(job_id: str, jd_text: str, temp_dir: str, file_names: list) -> None:
    """Start CV evaluation in a subprocess so it does not break uvicorn/asgiref executor."""
    set_running(job_id)
    proc = multiprocessing.Process(
        target=_cv_evaluation_worker_process,
        args=(job_id, jd_text, temp_dir, file_names, _cv_result_queue),
        daemon=True,
    )
    proc.start()

    timeout_sec = int(os.getenv("CV_EVAL_JOB_TIMEOUT_SEC", "900"))

    def _watchdog():
        proc.join(timeout_sec)
        job = get_job(job_id)
        if not job:
            return
        killed_after_deadline = False
        if proc.is_alive():
            killed_after_deadline = True
            logger.error(
                "CV eval job_id=%s exceeded %ss; terminating worker",
                job_id,
                timeout_sec,
            )
            proc.terminate()
            proc.join(20)
        # Allow the collector thread to dequeue after the worker exits.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            job = get_job(job_id)
            if job and job["status"] in ("completed", "failed"):
                return
            time.sleep(0.05)
        job = get_job(job_id)
        if job and job["status"] in ("pending", "running"):
            exitcode = proc.exitcode
            if killed_after_deadline:
                detail = (
                    f"The worker was still running after {timeout_sec}s and was terminated. "
                    "That usually means embeddings or OpenAI calls did not finish in time "
                    "(heavy PDFs, slow CPU, slow API, or a hang). "
                    "Check server logs for the last line from app.services.hr_service or "
                    "app.api.hr.cv_worker."
                )
            else:
                detail = (
                    "The worker exited before the API recorded a completed job "
                    f"(exit code {exitcode}). "
                    "If logs still show 'CV worker finished' / 'sending result', the result "
                    "often failed to cross the process boundary (multiprocessing queue). "
                    "Restart the server after updating to the SimpleQueue-based backend. "
                    "Otherwise inspect tracebacks from app.api.hr.cv_worker."
                )
            set_failed(job_id, detail)

    threading.Thread(target=_watchdog, daemon=True).start()


@bp.route('/cv/evaluate/async', methods=['POST'])
@require_auth
def evaluate_cvs_async():
    """
    Evaluate Multiple CVs (async).
    Returns job_id immediately; poll GET /cv/evaluate/status/<job_id> for result.
    """
    try:
        jd_text = request.form.get("job_description")
        if not jd_text:
            return error_response("job_description is required", status_code=400)
        cv_files = request.files.getlist("cv_files")
        if not cv_files or not any(f.filename for f in cv_files):
            return error_response("At least one CV file is required", status_code=400)

        temp_dir = tempfile.mkdtemp(prefix="cv_job_")
        file_names = []
        try:
            for i, f in enumerate(cv_files):
                name = secure_filename(f.filename) or f"file_{i}"
                if not name:
                    name = f"file_{i}"
                path = os.path.join(temp_dir, name)
                f.save(path)
                file_names.append(name)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return error_response(f"Failed to save uploads: {str(e)}", status_code=500)

        job_id = create_job()
        thread = threading.Thread(
            target=_run_cv_evaluation_async,
            args=(job_id, jd_text, temp_dir, file_names),
            daemon=True,
        )
        thread.start()

        return success_response(
            data={"job_id": job_id, "status": "pending"},
            message="CV evaluation started; poll /api/hr/cv/evaluate/status/<job_id> for result",
        )
    except Exception as e:
        return error_response(f"Error starting evaluation: {str(e)}", status_code=500)


@bp.route('/cv/evaluate/status/<job_id>', methods=['GET'])
@require_auth
def evaluate_cvs_status(job_id: str):
    """Get status and result of an async CV evaluation job."""
    try:
        job = get_job(job_id)
        if not job:
            return error_response("Job not found or expired", status_code=404)
        status = job["status"]
        logger.info("CV evaluate status poll: job_id=%s status=%s", job_id, status)
        if status == "completed" and job.get("result"):
            return success_response(data=job["result"], message="CV evaluation completed")
        if status == "failed":
            return success_response(
                data={"status": "failed", "error": job.get("error") or "Evaluation failed"},
                message=job.get("error") or "Evaluation failed",
            )
        return success_response(
            data={"job_id": job_id, "status": status},
            message="Job still in progress" if status in ("pending", "running") else status,
        )
    except Exception as e:
        logger.exception(
            "CV evaluate status error: job_id=%s error=%s",
            job_id,
            str(e),
            exc_info=True,
        )
        return error_response(f"Error fetching job status: {str(e)}", status_code=500)


@bp.route('/policy/upload', methods=['POST'])
@require_auth
def upload_policies():
    """
    Upload Policy Documents
    Upload multiple HR policy PDF documents
    ---
    tags:
      - HR AI Platform
    consumes:
      - multipart/form-data
    produces:
      - application/json
    security:
      - Bearer: []
    parameters:
      - in: formData
        name: policy_files
        type: array
        items:
          type: file
        collectionFormat: multi
        required: true
        description: Policy PDF files (multiple allowed)
    responses:
      200:
        description: Policies uploaded successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: 3 policy document(s) uploaded successfully
            data:
              type: object
              properties:
                message:
                  type: string
                document_count:
                  type: integer
                document_ids:
                  type: array
                  items:
                    type: integer
      400:
        description: Bad request
      401:
        description: Unauthorized
      500:
        description: Server error
    """
    try:
        policy_files = request.files.getlist('policy_files')
        if not policy_files or not any(f.filename for f in policy_files):
            return error_response("At least one policy file is required", status_code=400)
        
        user_id = g.user_id
        result = hr_service.upload_policies(policy_files, user_id)
        
        return success_response(data=result, message=result['message'])
    
    except Exception as e:
        return error_response(f"Error uploading policies: {str(e)}", status_code=500)


@bp.route('/policy/ask', methods=['POST'])
@require_auth
def ask_policy_question():
    """
    Ask Policy Question
    Ask a question about HR policies (uses uploaded policy documents)
    ---
    tags:
      - HR AI Platform
    consumes:
      - application/json
    produces:
      - application/json
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - question
          properties:
            question:
              type: string
              example: What is the leave policy for employees?
    responses:
      200:
        description: Policy question answered
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: Policy question answered
            data:
              type: object
              properties:
                answer:
                  type: string
                  example: According to the HR policy, employees are entitled to...
      401:
        description: Unauthorized
      422:
        description: Validation error
      500:
        description: Server error
    """
    try:
        question_data = PolicyQuestionRequest(**request.json)
    except ValidationError as e:
        errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return validation_error_response(errors)
    
    try:
        answer = hr_service.ask_policy_question(question_data.question)
        
        response_data = PolicyQuestionResponse(answer=answer)
        return success_response(data=response_data.dict(), message="Policy question answered")
    
    except Exception as e:
        return error_response(f"Error answering policy question: {str(e)}", status_code=500)


@bp.route('/technical/generate-questions', methods=['POST'])
@require_auth
@require_role('HR Manager')
def generate_technical_questions():
    """
    Generate Technical Questions
    Generate technical interview questions based on candidate CV and job description (HR Manager only)
    ---
    tags:
      - HR AI Platform
    consumes:
      - multipart/form-data
    produces:
      - application/json
    security:
      - Bearer: []
    parameters:
      - in: formData
        name: job_description
        type: string
        required: true
        description: Job description text
      - in: formData
        name: cv_file
        type: file
        required: true
        description: Candidate CV file (PDF or DOCX)
    responses:
      200:
        description: Technical questions generated successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: Technical questions generated
            data:
              type: object
              properties:
                questions:
                  type: array
                  items:
                    type: string
                  example: ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
      400:
        description: Bad request
      401:
        description: Unauthorized
      403:
        description: Forbidden (HR Manager role required)
      500:
        description: Server error
    """
    try:
        # Get job description from form data
        jd_text = request.form.get('job_description')
        if not jd_text:
            return error_response("job_description is required", status_code=400)
        
        # Get CV file
        cv_file = request.files.get('cv_file')
        if not cv_file or not cv_file.filename:
            return error_response("CV file is required", status_code=400)
        
        # Generate questions
        questions = hr_service.generate_technical_questions(cv_file, jd_text)
        
        response_data = TechnicalQuestionResponse(questions=questions)
        return success_response(data=response_data.dict(), message="Technical questions generated")
    
    except ValueError as e:
        return error_response(str(e), status_code=400)
    except Exception as e:
        return error_response(f"Error generating questions: {str(e)}", status_code=500)


@bp.route('/technical/evaluate-answers', methods=['POST'])
@require_auth
@require_role('HR Manager')
def evaluate_technical_answers():
    """
    Evaluate Technical Answers
    Evaluate candidate's answers to technical interview questions (HR Manager only)
    ---
    tags:
      - HR AI Platform
    consumes:
      - application/json
    produces:
      - application/json
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - questions
            - answers
          properties:
            questions:
              type: array
              items:
                type: string
              example: ["Question 1", "Question 2"]
            answers:
              type: array
              items:
                type: string
              example: ["Answer 1", "Answer 2"]
    responses:
      200:
        description: Technical evaluation completed
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: Technical evaluation completed
            data:
              type: object
              properties:
                evaluations:
                  type: array
                  items:
                    type: object
                total_score:
                  type: number
                  example: 75.0
                max_score:
                  type: number
                  example: 100.0
                overall_feedback:
                  type: string
      400:
        description: Bad request
      401:
        description: Unauthorized
      403:
        description: Forbidden (HR Manager role required)
      422:
        description: Validation error
      500:
        description: Server error
    """
    try:
        eval_data = TechnicalAnswerEvaluateRequest(**request.json)
    except ValidationError as e:
        errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return validation_error_response(errors)
    
    try:
        result = hr_service.evaluate_technical_answers(
            eval_data.questions,
            eval_data.answers
        )
        
        response_data = TechnicalAnswerEvaluateResponse(
            evaluations=result['evaluations'],
            total_score=result['total_score'],
            max_score=result['max_score'],
            overall_feedback=result['overall_feedback']
        )
        
        return success_response(data=response_data.dict(), message="Technical evaluation completed")
    
    except ValueError as e:
        return error_response(str(e), status_code=400)
    except Exception as e:
        return error_response(f"Error evaluating answers: {str(e)}", status_code=500)
