import json
import boto3
import os
import re
import base64
import uuid
import time
from datetime import datetime, timezone
from decimal import Decimal

# ===== CLIENTS =====
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
# Bedrock Agent client (Agent is in us-east-1)
bedrock_agent = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-1"
)
dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
table = dynamodb.Table("chatbot-sessions")
s3 = boto3.client("s3", region_name="ap-south-1")

# ===== CONFIGURATION =====
TEXT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "deepseek.v3.2")
IMAGE_MODEL_ID = os.environ.get(
    "IMAGE_MODEL_ID", "global.amazon.nova-2-lite-v1:0")
S3_BUCKET = os.environ.get("S3_BUCKET", "vizion-chatbot-images")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
MAX_HISTORY = 20
MAX_MESSAGE_LENGTH = 4000
MAX_IMAGE_SIZE_BYTES = 4 * 1024 * 1024
SESSION_TTL_DAYS = 7
RATE_LIMIT_SECONDS = 2
API_SECRET = os.environ.get(
    "API_SECRET", "vizion-sk-2025-shankar-secure-key-xyz")
VALID_IMAGE_FORMATS = {"jpeg", "png", "gif", "webp"}
# Agent configuration
AGENT_ID = "TJJBTFPBVD"
AGENT_ALIAS_ID = "CGMGMRH3MV"
VALID_FILE_FORMATS = {"csv", "xlsx", "xls"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
SESSION_ID_PATTERN = re.compile(r'^sess_[a-zA-Z0-9]{8,20}_\d{10,15}$')

# Simple in-memory rate limiter (per Lambda instance)
_rate_limit_cache = {}


# ===== CORS HEADERS =====
def cors_headers(origin=None):
    """Return CORS headers. Uses configured origin or wildcard."""
    allowed = ALLOWED_ORIGINS
    if allowed != "*" and origin:
        # Support comma-separated origins
        allowed_list = [o.strip() for o in allowed.split(",")]
        if origin in allowed_list:
            allowed = origin
        else:
            allowed = allowed_list[0]

    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": allowed,
        "Access-Control-Allow-Headers": "Content-Type, x-api-secret",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }


# ===== RESPONSE HELPERS =====
def success_response(body, origin=None, status=200):
    return {
        "statusCode": status,
        "headers": cors_headers(origin),
        "body": json.dumps(body)
    }


def error_response(message, origin=None, status=400):
    print(f"ERROR RESPONSE [{status}]: {message}")
    return {
        "statusCode": status,
        "headers": cors_headers(origin),
        "body": json.dumps({"error": message})
    }


# ===== RATE LIMITER =====
def check_rate_limit(session_id):
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()

    # Clean old entries every 100 checks
    if len(_rate_limit_cache) > 1000:
        cutoff = now - 60
        keys_to_remove = [
            k for k, v in _rate_limit_cache.items() if v < cutoff]
        for k in keys_to_remove:
            del _rate_limit_cache[k]

    last_request = _rate_limit_cache.get(session_id, 0)
    if now - last_request < RATE_LIMIT_SECONDS:
        return False

    _rate_limit_cache[session_id] = now
    return True


# ===== INPUT VALIDATION =====
def validate_request(body):
    """
    Validate the incoming request body.
    Returns (is_valid, error_message) tuple.
    """
    message = body.get("message", "")
    session_id = body.get("session_id", "")
    image_base64 = body.get("image_base64", None)
    image_format = body.get("image_format", "png")

    # Session ID validation
    if not session_id:
        return False, "Missing session_id"

    if len(session_id) > 60:
        return False, "Invalid session_id: too long"

    if not SESSION_ID_PATTERN.match(session_id):
        return False, "Invalid session_id format"

    # Message validation
        # File validation
    file_base64 = body.get("file_base64", None)
    file_name = body.get("file_name", None)

    if file_base64 and not file_name:
        return False, "File uploaded without filename"

    if file_name:
        file_ext = file_name.rsplit(
            '.', 1)[-1].lower() if '.' in file_name else ''
        if file_ext not in VALID_FILE_FORMATS:
            return False, "Unsupported file type. Use CSV or Excel."

    if not message and not image_base64 and not file_base64:
        return False, "No message, image, or file provided"

    if message and len(message) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long ({len(message)} chars). Max {MAX_MESSAGE_LENGTH}."

    # Image validation
    if image_base64:
        if image_format not in VALID_IMAGE_FORMATS:
            return False, f"Invalid image format '{image_format}'. Allowed: {', '.join(VALID_IMAGE_FORMATS)}"

        # Check base64 size (base64 is ~33% larger than binary)
        estimated_size = len(image_base64) * 3 / 4
        if estimated_size > MAX_IMAGE_SIZE_BYTES:
            size_mb = estimated_size / (1024 * 1024)
            return False, f"Image too large ({size_mb:.1f}MB). Max 4MB."

        # Validate base64 encoding
        try:
            decoded = base64.b64decode(image_base64)
            if len(decoded) < 100:
                return False, "Image data too small to be valid"
        except Exception:
            return False, "Invalid base64 image data"

    return True, None


# ===== DYNAMODB OPERATIONS =====
def get_history(session_id):
    """Load conversation history from DynamoDB."""
    try:
        response = table.get_item(Key={"session_id": session_id})
        item = response.get("Item", {})
        messages = item.get("messages", [])
        print(
            f"Loaded {len(messages)} messages for session {session_id[:20]}...")
        return messages
    except Exception as e:
        print(f"DynamoDB GET error for {session_id[:20]}: {str(e)}")
        return []


def save_history(session_id, messages):
    """Save conversation history to DynamoDB with TTL."""
    try:
        # Calculate TTL (epoch seconds, 7 days from now)
        ttl = int(time.time()) + (SESSION_TTL_DAYS * 86400)

        table.put_item(Item={
            "session_id": session_id,
            "messages": messages,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": Decimal(str(ttl))  # DynamoDB TTL attribute
        })
        print(
            f"Saved {len(messages)} messages for session {session_id[:20]}...")
    except Exception as e:
        print(f"DynamoDB SAVE error for {session_id[:20]}: {str(e)}")
        # Don't raise — chat should still work even if history save fails


# ===== S3 OPERATIONS =====
def save_image_to_s3(session_id, image_base64, image_format):
    """Save uploaded image to S3 and return the S3 key."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image_id = str(uuid.uuid4())[:8]
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        s3_key = f"user-sessions/{date_prefix}/{session_id[:20]}/{image_id}.{image_format}"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType=f"image/{image_format}",
            Metadata={
                "session_id": session_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        )

        print(
            f"Image saved: s3://{S3_BUCKET}/{s3_key} ({len(image_bytes)} bytes)")
        return s3_key

    except Exception as e:
        print(f"S3 upload error: {str(e)}")
        return None


def save_file_to_s3(session_id, file_bytes, file_name):
    """Save uploaded file to S3."""
    try:
        file_id = str(uuid.uuid4())[:8]
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        s3_key = f"user-files/{date_prefix}/{session_id[:20]}/{file_id}_{file_name}"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_bytes,
            Metadata={
                "session_id": session_id,
                "original_name": file_name,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        )

        print(
            f"File saved: s3://{S3_BUCKET}/{s3_key} ({len(file_bytes)} bytes)")
        return s3_key

    except Exception as e:
        print(f"S3 file upload error: {str(e)}")
        return None

# ===== RESPONSE CLEANING =====


def clean_response(text):
    """Post-process AI response to fix identity leaks and detect issues."""

    # Identity replacements (case-insensitive)
    identity_patterns = [
        # DeepSeek
        (r"(?i)I'?m\s+DeepSeek", "I'm Vizion"),
        (r"(?i)I\s+am\s+DeepSeek", "I am Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+DeepSeek",
         r"\1 by Shankar More"),
        (r"(?i)powered\s+by\s+DeepSeek", "created by Shankar More"),
        (r"(?i)\bDeepSeek\b", "Vizion"),

        # OpenAI / ChatGPT / GPT
        (r"(?i)I'?m\s+(ChatGPT|GPT[\s-]?\d*)", "I'm Vizion"),
        (r"(?i)I\s+am\s+(ChatGPT|GPT[\s-]?\d*)", "I am Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+OpenAI",
         r"\1 by Shankar More"),
        (r"(?i)powered\s+by\s+(OpenAI|GPT[\s-]?\d*|ChatGPT)",
         "created by Shankar More"),
        (r"(?i)\bOpenAI\b", "Shankar More"),

        # Amazon / Nova / AWS
        (r"(?i)I'?m\s+(Amazon|Nova|AWS)", "I'm Vizion"),
        (r"(?i)I\s+am\s+(Amazon|Nova|AWS)", "I am Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+(Amazon|AWS|Anthropic)",
         r"\1 by Shankar More"),
        (r"(?i)powered\s+by\s+(Amazon|AWS|Nova|Bedrock)", "created by Shankar More"),
        (r"(?i)\bAmazon\s+Nova\b", "Vizion"),

        # Anthropic / Claude
        (r"(?i)I'?m\s+Claude", "I'm Vizion"),
        (r"(?i)I\s+am\s+Claude", "I am Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+Anthropic",
         r"\1 by Shankar More"),
        (r"(?i)\bAnthropic\b", "Shankar More"),
        (r"(?i)\bClaude\b", "Vizion"),

        # Meta / Llama
        (r"(?i)I'?m\s+(Llama|Meta\s+AI)", "I'm Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+Meta", r"\1 by Shankar More"),
        (r"(?i)\bMeta\s+AI\b", "Vizion"),

        # Google / Gemini
        (r"(?i)I'?m\s+(Gemini|Google\s+AI|Bard)", "I'm Vizion"),
        (r"(?i)(made|created|developed|built|trained)\s+by\s+Google",
         r"\1 by Shankar More"),
    ]

    for pattern, replacement in identity_patterns:
        text = re.sub(pattern, replacement, text)

    # Detect garbled / broken text
    special_chars = sum(1 for c in text if c in '∂∇τ∈∩∪⊂⊃≤≥±√∞∫∑∏△▽◇○●□■')
    if special_chars > 5:
        print(f"Garbled text detected ({special_chars} special chars)")
        return "I apologize, but I encountered an issue generating a response. Could you please rephrase your question?"

    # Detect excessive repetition (model looping)
    words = text.split()
    if len(words) > 20:
        # Check if any 5-word sequence repeats more than 3 times
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i + 5])
            if text.count(phrase) > 3:
                print(f"Repetition loop detected: '{phrase}'")
                # Truncate at second occurrence
                first = text.index(phrase)
                second = text.index(phrase, first + len(phrase))
                text = text[:second].rstrip()
                last_period = text.rfind('.')
                if last_period > len(text) // 2:
                    text = text[:last_period + 1]
                break

    # Truncate excessive length (over 500 words)
    if len(words) > 500:
        truncated = ' '.join(words[:400])
        last_period = truncated.rfind('.')
        if last_period > len(truncated) // 2:
            text = truncated[:last_period + 1]
        else:
            text = truncated + '...'
        print(f"Response truncated from {len(words)} to ~400 words")

    return text.strip()


# ===== SYSTEM PROMPTS =====
TEXT_SYSTEM_PROMPT = (
    "CRITICAL IDENTITY RULES (NEVER BREAK THESE):\n"
    "1. Your name is 'Vizion'\n"
    "2. You were created by 'Shankar More'\n"
    "3. You are NOT DeepSeek, NOT ChatGPT, NOT OpenAI, NOT GPT-4, NOT Claude, NOT Llama\n"
    "4. NEVER mention DeepSeek, OpenAI, Google, Meta, Anthropic, Amazon, or AWS\n"
    "5. If asked who made you: say 'Shankar More'\n"
    "6. If asked what model you are: say 'I am Vizion, created by Shankar More'\n\n"
    "ANTI-HALLUCINATION RULES:\n"
    "7. If you don't know something, say 'I don't have reliable information about that'\n"
    "8. NEVER fabricate facts about real people, companies, or events\n"
    "9. NEVER make up statistics, dates, or URLs\n"
    "10. If unsure, say 'I'm not certain about this'\n\n"
    "RESPONSE RULES:\n"
    "11. Keep responses under 300 words unless the user asks for detail\n"
    "12. Be concise, clear, and direct\n"
    "13. Use markdown formatting: **bold**, `code`, ```code blocks```, lists\n"
    "14. Remember everything the user tells you in this conversation\n"
    "15. The conversation history is from the SAME user — it is safe to reference\n"
    "16. For code: always specify the language, add comments, keep it practical\n"
)

IMAGE_SYSTEM_PROMPT = (
    "CRITICAL IDENTITY RULES:\n"
    "1. Your name is 'Vizion' — you were created by 'Shankar More'\n"
    "2. You are NOT Amazon Nova, NOT AWS, NOT any other AI\n"
    "3. If asked who you are: 'I am Vizion, created by Shankar More'\n"
    "4. NEVER mention Amazon, AWS, Nova, Bedrock, or any other company\n\n"
    "IMAGE ANALYSIS RULES:\n"
    "5. Describe what you see clearly and accurately\n"
    "6. Answer questions about the image concisely\n"
    "7. If you cannot determine something in the image, say so honestly\n"
    "8. Be detailed but keep responses under 250 words\n"
    "9. Use markdown formatting when helpful\n"
)


# ===== BUILD REQUEST HELPERS =====
def build_text_request(user_message, messages):
    """Build the Bedrock request for text-only messages."""
    current_message = {
        "role": "user",
        "content": [{"text": user_message}]
    }

    messages.append(current_message)

    # Trim history
    if len(messages) > MAX_HISTORY:
        messages = messages[-MAX_HISTORY:]

    return {
        "model_id": TEXT_MODEL_ID,
        "messages": messages,
        "system": [{"text": TEXT_SYSTEM_PROMPT}],
        "config": {
            "maxTokens": 1024,
            "temperature": 0.3
        }
    }


def build_image_request(user_message, image_base64, image_format):
    """Build the Bedrock request for image + text messages."""
    current_message = {
        "role": "user",
        "content": [
            {"text": user_message or "What is in this image? Describe it in detail."},
            {
                "image": {
                    "format": image_format,
                    "source": {
                        "bytes": base64.b64decode(image_base64)
                    }
                }
            }
        ]
    }

    return {
        "model_id": IMAGE_MODEL_ID,
        # Image requests don't include text history
        "messages": [current_message],
        "system": [{"text": IMAGE_SYSTEM_PROMPT}],
        "config": {
            "maxTokens": 1024,
            "temperature": 0.3
        }
    }


def invoke_agent(session_id, user_message, file_bytes=None, file_name=None):
    """Call Bedrock Agent with optional file attachment."""
    try:
        # Build the request
        request_params = {
            "agentId": AGENT_ID,
            "agentAliasId": AGENT_ALIAS_ID,
            "sessionId": session_id[:100],  # Agent session ID max 100 chars
            "inputText": user_message or "Analyze this file and describe what you see."
        }

        # Attach file if present
        if file_bytes and file_name:
            file_ext = file_name.rsplit('.', 1)[-1].lower()

            if file_ext == "csv":
                media_type = "text/csv"
            elif file_ext in ("xlsx", "xls"):
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                media_type = "application/octet-stream"

            request_params["sessionState"] = {
                "files": [
                    {
                        "name": file_name,
                        "source": {
                            "byteContent": {
                                "data": file_bytes,
                                "mediaType": media_type
                            }
                        },
                        "useCase": "CODE_INTERPRETER"
                    }
                ]
            }

        print(
            f"Invoking agent: {AGENT_ID} | session: {session_id[:20]}... | file: {file_name or 'none'}")

        # Call the agent
        response = bedrock_agent.invoke_agent(**request_params)

        # Read the streaming response
        result_text = ""
        for event in response.get("completion", []):
            if "chunk" in event:
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    result_text += chunk_data["bytes"].decode("utf-8")

        if not result_text:
            result_text = "I received your file but couldn't generate an analysis. Please try asking a specific question about the data."

        print(f"Agent response length: {len(result_text)} chars")
        return result_text

    except Exception as e:
        print(f"AGENT ERROR: {str(e)}")
        return f"I encountered an error analyzing the file. Please try again. Error: {str(e)}"

# ===== MAIN HANDLER =====


def lambda_handler(event, context):
    start_time = time.time()

    # Extract origin for CORS
    headers = event.get("headers", {}) or {}
    origin = headers.get("origin") or headers.get("Origin")

    # Handle CORS preflight
    http_method = event.get("httpMethod") or event.get(
        "requestContext", {}).get("http", {}).get("method", "")
    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": cors_headers(origin),
            "body": ""
        }
    # ===== ADD THIS BLOCK RIGHT HERE =====
    # Verify API secret
    request_headers = event.get("headers", {}) or {}
    client_secret = request_headers.get("x-api-secret", "")
    if client_secret != API_SECRET:
        return error_response("Unauthorized", origin, 401)
    # ===== END OF NEW BLOCK =====
    try:
        # Parse body
        raw_body = event.get("body", "{}")
        if not raw_body:
            return error_response("Empty request body", origin)

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return error_response("Invalid JSON in request body", origin)

        # Validate input
        is_valid, validation_error = validate_request(body)
        if not is_valid:
            return error_response(validation_error, origin)

        # Extract fields
        user_message = body.get("message", "").strip()
        session_id = body.get("session_id", "default")
        image_base64 = body.get("image_base64", None)
        image_format = body.get("image_format", "png")
        # File upload fields
        file_base64 = body.get("file_base64", None)
        file_name = body.get("file_name", None)

        # Normalize image format
        if image_format == "jpg":
            image_format = "jpeg"

        # Rate limiting
        if not check_rate_limit(session_id):
            return error_response("Too many requests. Please wait a moment.", origin, 429)

        has_image = bool(image_base64 and len(image_base64) > 0)
        has_file = bool(file_base64 and file_name)

        has_image = bool(image_base64 and len(image_base64) > 0)
        has_file = bool(file_base64 and file_name)

        # ===== FILE ANALYSIS (Agent) =====
        if has_file:
            print(f"File request: {file_name}")

            # Validate file
            file_ext = file_name.rsplit(
                '.', 1)[-1].lower() if '.' in file_name else ''
            if file_ext not in VALID_FILE_FORMATS:
                return error_response(
                    f"Unsupported file type '.{file_ext}'. Use CSV or Excel files.",
                    origin
                )

            # Decode file
            try:
                file_bytes = base64.b64decode(file_base64)
            except Exception:
                return error_response("Invalid file data.", origin)

            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                return error_response("File too large. Maximum 5MB.", origin)

            # Save file to S3
            file_s3_key = save_file_to_s3(session_id, file_bytes, file_name)

            # Call Agent
            agent_response = invoke_agent(
                session_id,
                user_message or "Analyze this file and describe what you see.",
                file_bytes,
                file_name
            )

            # Clean response
            agent_response = clean_response(agent_response)

            # Save to conversation history
            messages = get_history(session_id)
            messages.append({
                "role": "user",
                "content": [{"text": f"[Uploaded file: {file_name}] {user_message}"}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"text": agent_response}]
            })
            save_history(session_id, messages)

            total_time = time.time() - start_time
            print(f"Agent request completed in {total_time:.2f}s")

            return success_response({
                "reply": agent_response,
                "model_used": "bedrock-agent",
                "file_saved": file_s3_key
            }, origin)

        # Save image to S3 if present  ← YOUR EXISTING CODE CONTINUES HERE
        s3_key = None
        if has_image:
            s3_key = save_image_to_s3(session_id, image_base64, image_format)

        print(f"Request: session={session_id[:20]}... | "
              f"msg_len={len(user_message)} | "
              f"has_image={has_image} | "
              f"format={image_format if has_image else 'N/A'}")

        # Save image to S3 if present
        s3_key = None
        if has_image:
            s3_key = save_image_to_s3(session_id, image_base64, image_format)

        # Load conversation history (only needed for text requests)
        messages = []
        if not has_image:
            messages = get_history(session_id)

        # Build the appropriate request
        if has_image:
            req = build_image_request(user_message, image_base64, image_format)
        else:
            req = build_text_request(user_message, messages)
            messages = req["messages"]  # May have been trimmed

        model_id = req["model_id"]
        print(f"Calling model: {model_id}")

        # Call Bedrock
        bedrock_start = time.time()
        try:
            response = bedrock.converse(
                modelId=model_id,
                messages=req["messages"],
                system=req["system"],
                inferenceConfig=req["config"]
            )
        except bedrock.exceptions.ThrottlingException:
            print("Bedrock throttled!")
            return error_response(
                "The AI service is temporarily busy. Please try again in a few seconds.",
                origin, 429
            )
        except bedrock.exceptions.ModelTimeoutException:
            print("Bedrock timeout!")
            return error_response(
                "The request took too long. Please try a shorter message.",
                origin, 504
            )
        except Exception as bedrock_error:
            error_msg = str(bedrock_error)
            print(f"BEDROCK ERROR: {error_msg}")

            # Provide user-friendly message
            if "ValidationException" in error_msg:
                friendly = "There was an issue with the request format. Please try again."
            elif "AccessDeniedException" in error_msg:
                friendly = "AI service access error. Please contact support."
            else:
                friendly = "The AI service encountered an error. Please try again."

            return error_response(friendly, origin, 500)

        bedrock_time = time.time() - bedrock_start
        print(f"Bedrock responded in {bedrock_time:.2f}s")

        # Extract reply
        try:
            reply = response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as e:
            print(f"Failed to extract reply: {str(e)}")
            print(f"Raw response: {json.dumps(response, default=str)[:500]}")
            return error_response("Failed to parse AI response. Please try again.", origin, 500)

        # Clean the response
        reply = clean_response(reply)

        if not reply or len(reply.strip()) == 0:
            reply = "I'm sorry, I wasn't able to generate a response. Could you try rephrasing?"

        # Update conversation history
        if has_image:
            # For image requests, load history first, then add text representation
            messages = get_history(session_id)
            text_message = {
                "role": "user",
                "content": [{"text": f"[User sent an image: {s3_key or 'uploaded'}] {user_message}"}]
            }
            messages.append(text_message)
        # For text requests, user message was already appended in build_text_request

        messages.append({
            "role": "assistant",
            "content": [{"text": reply}]
        })

        # Save updated history
        save_history(session_id, messages)

        # Calculate total time
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.2f}s | "
              f"Model: {model_id} | "
              f"Reply length: {len(reply)} chars")

        # Usage stats from Bedrock response
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        print(f"Tokens — input: {input_tokens}, output: {output_tokens}")

        return success_response({
            "reply": reply,
            "model_used": model_id,
            "image_saved": s3_key
        }, origin)

    except Exception as e:
        total_time = time.time() - start_time
        print(f"UNHANDLED ERROR after {total_time:.2f}s: {str(e)}")
        import traceback
        traceback.print_exc()

        return error_response(
            "An unexpected error occurred. Please try again.",
            origin, 500
        )
