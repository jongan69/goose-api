from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from services.pdf_parser import extract_invoice_data
from models.invoice import InvoiceData
import subprocess
import re
import os
import json
import logging
import sys
import secrets

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("goose-api")

# Security
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", secrets.token_urlsafe(32))
api_key_header = APIKeyHeader(name=API_KEY_NAME)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

# Initialize FastAPI app
app = FastAPI(
    title="Next Invoice Parser API",
    description="API for extracting structured data from invoice PDFs",
    version="1.0.0"
)

# Add CORS middleware with proper settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Add rate limiting middleware
from fastapi import Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
            
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

class TaskGuidelines(BaseModel):
    """Guidelines for specific tasks."""
    credit_assessment: str = """
    Analyze the invoice and provide a credit assessment with the following structure:
    {
        "credit_score": number,  # 0-1000
        "risk_level": string,    # "low", "medium", "high"
        "confidence": number,    # 0-100
        "factors": [string],     # List of factors considered
        "recommendation": string # Brief recommendation
    }
    """
    
    lender_matching: str = """
    Analyze the invoice and credit assessment to provide lender matches:
    {
        "matches": [
            {
                "lender_name": string,
                "match_score": number,  # 0-100
                "advance_rate": number,  # 0-100
                "terms": string,
                "reasoning": string
            }
        ],
        "best_match": string,  # Name of best matching lender
        "confidence": number   # 0-100
    }
    """
    
    funding_recommendation: str = """
    Provide a funding recommendation based on the invoice and credit assessment:
    {
        "recommended_advance_rate": number,  # 0-100
        "estimated_funding_time": string,
        "terms": {
            "duration": string,
            "fees": string,
            "conditions": [string]
        },
        "confidence": number,  # 0-100
        "rationale": string
    }
    """

class GooseInput(BaseModel):
    instructions: str
    session_name: str = "api-session"
    task: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class GooseResponse(BaseModel):
    raw_response: str
    parsed_actions: List[Dict[str, Any]]
    error: Optional[str] = None
    shell_command: Optional[str] = None
    shell_response: Optional[str] = None
    task_response: Optional[Dict[str, Any]] = None

class ParseInvoiceRequest(BaseModel):
    """Request model for invoice parsing with instructions."""
    instructions: str = "Analyze this invoice and extract all relevant information."

class ParseInvoiceResponse(BaseModel):
    """Response model for invoice parsing endpoint."""
    invoice_data: InvoiceData
    goose_response: GooseResponse

class CreditAssessmentResponse(BaseModel):
    """Response model for credit assessment.
    
    TypeScript interface:
    ```typescript
    interface CreditAssessment {
        credit_score: number;      // 0-1000
        risk_level: 'low' | 'medium' | 'high';
        confidence: number;        // 0-100
        factors: string[];         // List of factors considered
        recommendation: string;    // Brief recommendation
        error?: string;           // Optional error message
    }
    ```
    """
    credit_score: int
    risk_level: str
    confidence: int
    factors: List[str]
    recommendation: str
    error: Optional[str] = None

class LenderMatch(BaseModel):
    """Model for a lender match."""
    lender_name: str
    match_score: int
    advance_rate: float
    terms: str
    reasoning: str

class LenderMatchingResponse(BaseModel):
    """Response model for lender matching."""
    matches: List[LenderMatch]
    best_match: str
    confidence: int
    error: Optional[str] = None

class FundingTerms(BaseModel):
    """Model for funding terms."""
    duration: str
    fees: str
    conditions: List[str]

class FundingRecommendation(BaseModel):
    """Response model for funding recommendation."""
    recommended_advance_rate: float
    estimated_funding_time: str
    terms: FundingTerms
    confidence: int
    rationale: str
    error: Optional[str] = None

def extract_shell_command(text: str) -> dict:
    """Extract shell command and its output from the response."""
    command_pattern = r'command: (.*?)(?:\n\n|\n$)'
    command_match = re.search(command_pattern, text, re.DOTALL)
    
    if command_match:
        command = command_match.group(1).strip()
        response_text = text[text.find(command) + len(command):].strip()
        return {
            "command": command,
            "response": response_text
        }
    return None

def clean_goose_output(output: str) -> str:
    if not output:
        logger.warning("Received empty output from goose command")
        return ""
        
    logger.debug(f"Raw goose output: {output}")
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)

    filtered_lines = []
    for line in clean_output.splitlines():
        if not any(kw in line.lower() for kw in ["logging to", "working directory", "starting session"]):
            filtered_lines.append(line.strip())

    cleaned = "\n".join(filtered_lines).strip()
    logger.debug(f"Cleaned output: {cleaned}")
    return cleaned

def parse_goose_response(response: str, task: Optional[str] = None) -> dict:
    if not response:
        logger.warning("Received empty response to parse")
        return {
            "raw_response": "",
            "parsed_actions": [],
            "error": "Empty response received from goose command"
        }
        
    logger.debug(f"Parsing response: {response}")
    
    shell_info = extract_shell_command(response)
    if shell_info:
        return {
            "raw_response": response,
            "parsed_actions": [{
                "action": "shell_command",
                "parameters": shell_info
            }],
            "shell_command": shell_info["command"],
            "shell_response": shell_info["response"]
        }
    
    function_pattern = r'<function=([^{]+){([^}]+)}</function>'
    matches = re.finditer(function_pattern, response)
    
    parsed_actions = []
    for match in matches:
        function_name = match.group(1)
        try:
            params = json.loads(match.group(2))
            parsed_actions.append({
                "action": function_name,
                "parameters": params
            })
            logger.debug(f"Parsed action: {function_name} with params: {params}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for function {function_name}: {e}")
            parsed_actions.append({
                "action": function_name,
                "parameters": match.group(2)
            })
    
    result = {
        "raw_response": response,
        "parsed_actions": parsed_actions
    }
    
    if not parsed_actions:
        logger.warning("No actions were parsed from the response")
        result["error"] = "No actions found in response"
        
    logger.debug(f"Final parsed result: {json.dumps(result)}")
    return result

@app.post("/run-goose/", response_model=GooseResponse)
def run_goose(input: GooseInput):
    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logger.info(f"[{request_id}] Received request - Session: {input.session_name}")
    logger.debug(f"[{request_id}] Instructions: {input.instructions}")
    
    try:
        # Prepare the instruction file with task guidelines if specified
        instruction_content = input.instructions
        if input.task and hasattr(TaskGuidelines, input.task):
            task_guidelines = getattr(TaskGuidelines, input.task)
            instruction_content = f"{input.instructions}\n\nTask Guidelines:\n{task_guidelines}"
            
        instruction_file = "/tmp/goose_input.txt"
        with open(instruction_file, "w") as f:
            f.write(instruction_content)
        logger.debug(f"[{request_id}] Wrote instructions to {instruction_file}")

        env = os.environ.copy()
        env["GOOSE_NO_KEYRING"] = "1"
        env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/0/bus"
        env["RUST_BACKTRACE"] = "1"
        
        try:
            # First, ensure we have a session
            logger.info(f"[{request_id}] Creating/Resuming session: {input.session_name}")
            session_result = subprocess.run(
                ["goose", "session", "--name", input.session_name],
                capture_output=True,
                text=True,
                env=env
            )
            
            if session_result.returncode != 0:
                logger.error(f"[{request_id}] Failed to create/resume session: {session_result.stderr}")
                raise HTTPException(status_code=500, detail=f"Failed to create/resume session: {session_result.stderr}")

            # Now run the command in the session
            logger.info(f"[{request_id}] Executing goose command in session")
            result = subprocess.run(
                ["goose", "run", "-i", instruction_file, "--name", input.session_name, "--debug"],
                capture_output=True,
                text=True,
                env=env
            )

            logger.debug(f"[{request_id}] Command return code: {result.returncode}")
            logger.debug(f"[{request_id}] Command stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"[{request_id}] Command stderr: {result.stderr}")

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                logger.error(f"[{request_id}] Goose command failed: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

            if not result.stdout:
                logger.warning(f"[{request_id}] Empty stdout from goose command")
                return {
                    "raw_response": "",
                    "parsed_actions": [],
                    "error": "No response received from goose command"
                }

            cleaned_stdout = clean_goose_output(result.stdout)
            parsed_response = parse_goose_response(cleaned_stdout, input.task)
            
            # Try to extract structured task response if task was specified
            if input.task:
                try:
                    # Look for JSON in the response
                    json_pattern = r'({[\s\S]*})'
                    json_match = re.search(json_pattern, cleaned_stdout)
                    if json_match:
                        task_response = json.loads(json_match.group(1))
                        parsed_response["task_response"] = task_response
                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] Failed to parse task response as JSON")
            
            logger.info(f"[{request_id}] Successfully processed request")
            return parsed_response

        except Exception as e:
            logger.error(f"[{request_id}] Error executing goose command: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse-invoice/", response_model=ParseInvoiceResponse, dependencies=[Depends(get_api_key)])
async def parse_invoice(
    file: UploadFile = File(...),
    request: ParseInvoiceRequest = None
):
    """
    Parse an invoice PDF and return structured data.
    
    This endpoint accepts a PDF file upload and extracts key invoice details
    including invoice number, dates, amounts, and parties involved.
    
    Args:
        file: The PDF file to parse (required, sent as form-data with key 'file')
        request: Optional request body containing instructions and session_name
                (sent as form-data with key 'request' as JSON string)
    
    Example curl request:
        curl -X POST \\
          -F "file=@/path/to/invoice.pdf" \\
          -F "request={\\"instructions\\": \\"Analyze this invoice\\", \\"session_name\\": \\"my-session\\"}" \\
          http://localhost:8000/parse-invoice/
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    try:
        contents = await file.read()
        invoice_data = extract_invoice_data(contents)
        
        # Direct goose integration
        goose_input = GooseInput(
            instructions=f"""
            Analyze the following invoice data and provide insights:
            
            Invoice Number: {invoice_data.invoice_number}
            Issue Date: {invoice_data.issue_date}
            Due Date: {invoice_data.due_date}
            Total Amount: {invoice_data.total_amount} {invoice_data.currency}
            Buyer: {invoice_data.buyer.name}
            Confidence Score: {invoice_data.confidence_score}
            
            Please analyze this data and provide:
            1. A summary of the invoice
            2. Any potential risks or concerns
            3. Recommendations for processing this invoice
            """,
            session_name=request.session_name if request else "local-goose-session",
            data={
                "invoice_data": invoice_data.dict(),
                "task": "invoice_analysis"
            }
        )
        response = run_goose(goose_input)
        print(response)
        
        return {
            "invoice_data": invoice_data,
            "goose_response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")

@app.post("/credit-assessment/", response_model=CreditAssessmentResponse)
async def assess_credit(
    invoice: Dict[str, Any],
    session_name: Optional[str] = None
):
    """
    Assess the credit risk of an invoice.
    
    Args:
        invoice: The invoice data to assess
        session_name: Optional session name for the goose agent
    
    Returns:
        Credit assessment including score, risk level, and factors
    """
    try:
        goose_input = GooseInput(
            instructions="""
            Analyze this invoice and provide a credit assessment with the following structure:
            {
                "credit_score": number,  # 0-1000
                "risk_level": string,    # "low", "medium", "high"
                "confidence": number,    # 0-100
                "factors": [string],     # List of factors considered
                "recommendation": string # Brief recommendation
            }
            
            Consider the following factors:
            1. Invoice amount and currency
            2. Payment terms and due date
            3. Buyer's credit history
            4. Industry risk factors
            5. Market conditions
            """,
            session_name=session_name or f"invoice-{invoice.get('id', 'unknown')}",
            task="credit_assessment",
            data={"invoice": invoice}
        )
        
        response = run_goose(goose_input)
        
        # Check if response is a dictionary and has an error key
        if isinstance(response, dict) and response.get("error"):
            return CreditAssessmentResponse(
                credit_score=0,
                risk_level="unknown",
                confidence=0,
                factors=[],
                recommendation="Unable to assess credit",
                error=response["error"]
            )
            
        # Check if we have a task response
        if isinstance(response, dict) and response.get("task_response"):
            return CreditAssessmentResponse(**response["task_response"])
            
        # Try to parse the raw response if no structured response
        if isinstance(response, dict) and response.get("raw_response"):
            try:
                # Look for JSON in the response
                json_pattern = r'({[\s\S]*})'
                json_match = re.search(json_pattern, response["raw_response"])
                if json_match:
                    parsed_response = json.loads(json_match.group(1))
                    return CreditAssessmentResponse(**parsed_response)
            except (json.JSONDecodeError, KeyError):
                pass
                
        # If we get here, we couldn't parse a valid response
        return CreditAssessmentResponse(
            credit_score=0,
            risk_level="unknown",
            confidence=0,
            factors=[],
            recommendation="Unable to parse response",
            error="No structured response received"
        )
        
    except Exception as e:
        logger.error(f"Error in credit assessment: {str(e)}", exc_info=True)
        return CreditAssessmentResponse(
            credit_score=0,
            risk_level="unknown",
            confidence=0,
            factors=[],
            recommendation="Error processing request",
            error=str(e)
        )

@app.post("/lender-matching/", response_model=LenderMatchingResponse, dependencies=[Depends(get_api_key)])
async def match_lenders(
    invoice: Dict[str, Any],
    credit_assessment: Dict[str, Any],
    session_name: Optional[str] = None
):
    """
    Find matching lenders for an invoice based on credit assessment.
    
    Args:
        invoice: The invoice data
        credit_assessment: The credit assessment data
        session_name: Optional session name for the goose agent
    
    Returns:
        List of matching lenders with their terms
    """
    try:
        goose_input = GooseInput(
            instructions="""
            Based on the invoice and credit assessment, provide lender matches with the following structure:
            {
                "matches": [
                    {
                        "lender_name": string,
                        "match_score": number,  # 0-100
                        "advance_rate": number,  # 0-100
                        "terms": string,
                        "reasoning": string
                    }
                ],
                "best_match": string,  # Name of best matching lender
                "confidence": number   # 0-100
            }
            
            Consider the following factors:
            1. Credit score and risk level
            2. Invoice amount and currency
            3. Industry and market conditions
            4. Payment terms and due date
            5. Historical performance
            """,
            session_name=session_name or f"invoice-{invoice.get('id', 'unknown')}",
            task="lender_matching",
            data={
                "invoice": invoice,
                "credit_assessment": credit_assessment
            }
        )
        
        response = run_goose(goose_input)
        
        if response.error:
            return LenderMatchingResponse(
                matches=[],
                best_match="",
                confidence=0,
                error=response.error
            )
            
        if response.task_response:
            return LenderMatchingResponse(**response.task_response)
            
        # Try to parse the raw response if no structured response
        try:
            json_pattern = r'({[\s\S]*})'
            json_match = re.search(json_pattern, response.raw_response)
            if json_match:
                parsed_response = json.loads(json_match.group(1))
                return LenderMatchingResponse(**parsed_response)
        except (json.JSONDecodeError, KeyError):
            pass
            
        return LenderMatchingResponse(
            matches=[],
            best_match="",
            confidence=0,
            error="No structured response received"
        )
        
    except Exception as e:
        logger.error(f"Error in lender matching: {str(e)}", exc_info=True)
        return LenderMatchingResponse(
            matches=[],
            best_match="",
            confidence=0,
            error=str(e)
        )

@app.post("/funding-options/", response_model=FundingRecommendation, dependencies=[Depends(get_api_key)])
async def get_funding_options(
    invoice: Dict[str, Any],
    credit_assessment: Dict[str, Any],
    selected_lender: Dict[str, Any],
    session_name: Optional[str] = None
):
    """
    Get funding options for an invoice based on credit assessment and selected lender.
    
    Args:
        invoice: The invoice data
        credit_assessment: The credit assessment data
        selected_lender: The selected lender data
        session_name: Optional session name for the goose agent
    
    Returns:
        Funding recommendation with terms and conditions
    """
    try:
        goose_input = GooseInput(
            instructions="""
            Based on the invoice, credit assessment, and selected lender, provide funding options with the following structure:
            {
                "recommended_advance_rate": number,  # 0-100
                "estimated_funding_time": string,
                "terms": {
                    "duration": string,
                    "fees": string,
                    "conditions": [string]
                },
                "confidence": number,  # 0-100
                "rationale": string
            }
            
            Consider the following factors:
            1. Credit score and risk level
            2. Lender's terms and conditions
            3. Invoice amount and currency
            4. Market conditions
            5. Historical performance
            """,
            session_name=session_name or f"invoice-{invoice.get('id', 'unknown')}",
            task="funding_recommendation",
            data={
                "invoice": invoice,
                "credit_assessment": credit_assessment,
                "selected_lender": selected_lender
            }
        )
        
        response = run_goose(goose_input)
        
        if response.error:
            return FundingRecommendation(
                recommended_advance_rate=0,
                estimated_funding_time="unknown",
                terms=FundingTerms(duration="", fees="", conditions=[]),
                confidence=0,
                rationale="Unable to provide funding options",
                error=response.error
            )
            
        if response.task_response:
            return FundingRecommendation(**response.task_response)
            
        # Try to parse the raw response if no structured response
        try:
            json_pattern = r'({[\s\S]*})'
            json_match = re.search(json_pattern, response.raw_response)
            if json_match:
                parsed_response = json.loads(json_match.group(1))
                return FundingRecommendation(**parsed_response)
        except (json.JSONDecodeError, KeyError):
            pass
            
        return FundingRecommendation(
            recommended_advance_rate=0,
            estimated_funding_time="unknown",
            terms=FundingTerms(duration="", fees="", conditions=[]),
            confidence=0,
            rationale="Unable to parse response",
            error="No structured response received"
        )
        
    except Exception as e:
        logger.error(f"Error in funding options: {str(e)}", exc_info=True)
        return FundingRecommendation(
            recommended_advance_rate=0,
            estimated_funding_time="unknown",
            terms=FundingTerms(duration="", fees="", conditions=[]),
            confidence=0,
            rationale="Error processing request",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "next-invoice-parser"}