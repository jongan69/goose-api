from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import subprocess
import re
import os
import json
import logging
import sys
from datetime import datetime

from services.pdf_parser import extract_invoice_data
from models.invoice import InvoiceData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("goose-api")

# Initialize FastAPI app
app = FastAPI(
    title="Next Invoice Parser API",
    description="API for extracting structured data from invoice PDFs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/parse-invoice/", response_model=ParseInvoiceResponse)
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "next-invoice-parser"}