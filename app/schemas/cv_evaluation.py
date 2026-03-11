from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SkillScore(BaseModel):
    """Skill score schema"""
    skill: str
    score: float = Field(..., ge=0, le=100, description="Score from 0-100")


class SkillStatus(BaseModel):
    """Skill status schema"""
    missing: List[str] = Field(default_factory=list, description="Skills mentioned in JD but weak in CV")
    absent: List[str] = Field(default_factory=list, description="Skills required in JD but missing from CV")
    strong: List[str] = Field(default_factory=list, description="Skills that are strong in CV")


class HireRecommendation(BaseModel):
    """Hire recommendation schema"""
    recommendation: str = Field(..., description="Strong Hire, Consider, or Not Recommended")
    emoji: str = Field(..., description="Recommendation emoji")
    color: str = Field(..., description="Recommendation color code")
    confidence: float = Field(..., ge=0, le=100, description="Confidence score")
    risk_level: str = Field(..., description="Low, Medium, or High")


class CVResult(BaseModel):
    """CV evaluation result schema"""
    name: str = Field(..., description="CV filename")
    score: float = Field(..., ge=0, le=100, description="Similarity score")
    evaluation: str = Field(..., description="Detailed evaluation text")
    skill_scores: Dict[str, float] = Field(default_factory=dict, description="Skill scores per category")
    skill_status: SkillStatus = Field(..., description="Skill status breakdown")
    hire_recommendation: HireRecommendation = Field(..., description="Hire recommendation")
    skills: Optional[List[str]] = Field(default=None, description="List of skill categories extracted from job description")


class CVEvaluationRequest(BaseModel):
    """CV evaluation request schema"""
    job_description: str = Field(..., min_length=1, description="Job description text")
    # Note: Files will be handled separately in the endpoint
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "We are looking for a Python developer with Flask experience..."
            }
        }


class CVEvaluationResponse(BaseModel):
    """CV evaluation response schema"""
    results: List[CVResult] = Field(..., description="Ranked list of CV evaluation results")
    executive_kpis: Dict[str, Any] = Field(..., description="Executive KPIs (total candidates, average match, etc.)")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time in seconds (for reference)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [],
                "executive_kpis": {
                    "total_candidates": 5,
                    "average_match": 75.5,
                    "top_score": 90.0,
                    "top_5_count": 5
                }
            }
        }
