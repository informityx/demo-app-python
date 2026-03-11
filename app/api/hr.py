from flask import Blueprint, request, g
from pydantic import ValidationError
from app.middleware.auth import require_auth, require_role
from app.services.hr_service import HRService
from app.utils.response import success_response, error_response, validation_error_response
from app.schemas.cv_evaluation import CVEvaluationRequest, CVEvaluationResponse
from app.schemas.policy import PolicyUploadRequest, PolicyQuestionRequest, PolicyQuestionResponse
from app.schemas.technical import (
    TechnicalQuestionGenerateRequest, TechnicalQuestionResponse,
    TechnicalAnswerEvaluateRequest, TechnicalAnswerEvaluateResponse
)

bp = Blueprint('hr', __name__)
hr_service = HRService()


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
        
        # Evaluate CVs
        result = hr_service.evaluate_cvs(cv_files, jd_text)
        
        # Format response
        response_data = CVEvaluationResponse(
            results=result['results'],
            executive_kpis=result['executive_kpis'],
            processing_time_seconds=result.get('processing_time_seconds'),
        )
        
        return success_response(data=response_data.dict(), message="CV evaluation completed")
    
    except Exception as e:
        return error_response(f"Error evaluating CVs: {str(e)}", status_code=500)


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
