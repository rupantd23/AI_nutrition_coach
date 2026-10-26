import base64
import io
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from sarvamai import SarvamAI


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DICTATION_LANGUAGES = {
    "unknown": "Auto-detect",
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
    "kok-IN": "Konkani",
    "ks-IN": "Kashmiri",
    "sd-IN": "Sindhi",
    "sa-IN": "Sanskrit",
    "sat-IN": "Santali",
    "mni-IN": "Manipuri",
    "brx-IN": "Bodo",
    "mai-IN": "Maithili",
    "doi-IN": "Dogri",
}
TTS_LANGUAGES = {
    "en-IN": {"name": "English", "speaker": "aditi_en_stories"},
    "hi-IN": {"name": "Hindi", "speaker": "roopa_hi_narration"},
    "bn-IN": {"name": "Bengali", "speaker": "roopa_bn_conversational"},
    "gu-IN": {"name": "Gujarati", "speaker": "pooja_gu_conversational"},
    "kn-IN": {"name": "Kannada", "speaker": "chaitra_kn_narration"},
    "mr-IN": {"name": "Marathi", "speaker": "mrunal_mr_narration"},
    "pa-IN": {"name": "Punjabi", "speaker": "harpreet_pa_narration"},
    "ta-IN": {"name": "Tamil", "speaker": "gokul_ta_narration"},
    "te-IN": {"name": "Telugu", "speaker": "kavitha_te_narration"},
}


def load_env_file():
    """Load simple KEY=value entries without adding a dotenv dependency."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name:
            os.environ.setdefault(name, value)


load_env_file()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.context_processor
def add_tts_languages():
    return {"tts_languages": TTS_LANGUAGES}

DIETARY_OPTIONS = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "gluten_free": "gluten-free",
    "dairy_free": "dairy-free",
    "nut_free": "peanut and tree-nut free",
    "low_calorie": "low-calorie",
    "high_protein": "higher-protein",
}

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "servings": {"type": "integer"},
                    "prep_minutes": {"type": "integer"},
                    "cook_minutes": {"type": "integer"},
                    "total_minutes": {"type": "integer"},
                    "calories_per_serving": {"type": "integer"},
                    "protein_g": {"type": "number"},
                    "carbs_g": {"type": "number"},
                    "fat_g": {"type": "number"},
                    "fiber_g": {"type": "number"},
                    "sugar_g": {"type": "number"},
                    "sodium_mg": {"type": "integer"},
                    "why_it_matches": {"type": "string"},
                    "nutrition_note": {"type": "string"},
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "substitution_note": {"type": "string"},
                },
                "required": [
                    "name", "description", "servings", "prep_minutes", "cook_minutes",
                    "total_minutes", "calories_per_serving", "protein_g", "carbs_g", "fat_g",
                    "fiber_g", "sugar_g", "sodium_mg", "why_it_matches", "nutrition_note",
                    "ingredients", "steps", "tags", "substitution_note",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recipes"],
    "additionalProperties": False,
}


def sarvam_client():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("Add SARVAM_API_KEY to .env, then restart the app.")
    return SarvamAI(api_subscription_key=api_key)


def identify_food(client, image_bytes, mime_type):
    data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    response = client.chat.completions_v2(
        model="gemma4",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identify the visible prepared food and likely ingredients in this image. "
                            "Be concise, mention uncertainty, and do not claim exact portion sizes or "
                            "nutrition. If it is not food, say that clearly."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content or "Food could not be identified with confidence."


def build_recipe_request(client, source_text, restrictions, custom_restrictions, max_calories, discovery_mode=False):
    rules = [f"Dietary preferences and exclusions: {', '.join(restrictions) or 'none specified'}." ]
    if custom_restrictions:
        rules.append(f"Additional user restrictions: {custom_restrictions}.")
    if max_calories:
        rules.append(f"Aim for no more than {max_calories} estimated calories per serving where practical.")
    rules.append(
        "Treat every stated allergy or exclusion as mandatory. Do not include an excluded ingredient; "
        "if the request cannot be met safely, return an empty recipes list and explain no alternative in the recipe description."
    )
    rules.append(
        "Return two practical recipe ideas when possible. Include a short reason each fits the user's "
        "preferences, a rough prep time, cook time, total minutes, complete ingredients with quantities, "
        "concise cooking steps, tags, and estimated per-serving calories, protein, carbohydrates, fat, "
        "fiber, sugar, and sodium. Clearly label every nutrition value as an estimate. Prefer balanced "
        "meals and do not provide medical advice."
    )
    ingredient_rule = (
        "The user is choosing what to eat and has no ingredients selected. Suggest accessible Indian "
        "recipes and provide a complete shopping list with quantities."
        if discovery_mode else
        "Use the listed ingredients and common pantry basics; clearly label any extra pantry basics."
    )
    system_message = (
        "You are a practical Indian recipe assistant. Follow the user's food restrictions exactly. "
        + ingredient_rule + " Keep cooking times realistic. " + " ".join(rules)
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": source_text},
    ]
    response = client.chat.completions(
        model="sarvam-105b",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "recipe_recommendations",
                "strict": True,
                "schema": RECIPE_SCHEMA,
            },
        },
        max_tokens=3500,
        reasoning_effort=None,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return parse_recipe_json(content)
    except (json.JSONDecodeError, ValueError):
        # Recover once if a response is cut off or fails the expected structure.
        recovery_messages = [
            {
                "role": "system",
                "content": (
                    system_message
                    + " Return only one valid JSON object with this exact structure; no markdown or prose: "
                    + json.dumps(RECIPE_SCHEMA, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": source_text},
        ]
        retry = client.chat.completions(
            model="sarvam-105b",
            messages=recovery_messages,
            response_format={"type": "json_object"},
            max_tokens=3500,
            reasoning_effort=None,
        )
        return parse_recipe_json(retry.choices[0].message.content or "{}")


def parse_recipe_json(content):
    """Accept a JSON object even when the model wraps it in a Markdown fence or prose."""
    if isinstance(content, dict):
        result = content
    else:
        text = str(content).strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text[:4].lower() == "json":
                text = text[4:].strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Find the first complete JSON object while respecting braces in strings.
            start = text.find("{")
            if start < 0:
                raise
            depth = 0
            in_string = False
            escaped = False
            end = None
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                raise
            result = json.loads(text[start:end])
    if not isinstance(result, dict) or not isinstance(result.get("recipes"), list):
        raise ValueError("Sarvam returned recipe data in an unexpected format.")
    return result


def api_error_message(error):
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    provider_error = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(provider_error, str):
        detail = provider_error
        code = None
        request_id = None
    elif isinstance(provider_error, dict):
        detail = provider_error.get("message") or ""
        code = provider_error.get("code")
        request_id = provider_error.get("request_id")
    else:
        detail = ""
        code = None
        request_id = None

    if status == 402 or code == "insufficient_quota_error":
        message = "Sarvam has no credits available. Add credits to the account and try again."
    elif status == 403 or code == "invalid_api_key_error":
        message = "Sarvam rejected the API key. Check SARVAM_API_KEY in .env, then restart the app."
    elif status == 429 or code == "rate_limit_exceeded_error":
        message = "Sarvam rate limit reached. Wait a moment and retry."
    elif status or code:
        label = f"HTTP {status}" if status else "Sarvam API error"
        message = f"{label}{f' ({code})' if code else ''}: {detail or 'request was rejected'}"
    elif isinstance(error, json.JSONDecodeError):
        message = "Sarvam replied, but the recipe response was not valid JSON. Retry the request."
    else:
        message = f"Could not reach or parse Sarvam ({type(error).__name__}). Check the app terminal for details."

    if request_id:
        message += f" Request ID: {request_id}."
    return message[:500]


@app.get("/")
def home():
    return render_template("index.html", dietary_options=DIETARY_OPTIONS)


@app.post("/recommend")
def recommend():
    mode = request.form.get("mode", "ingredients")
    image = request.files.get("image")
    ingredients = request.form.get("ingredients", "").strip()
    if not image or not image.filename:
        image = None
    if mode == "photo" and image is None and not request.form.get("photo_context", "").strip():
        return render_template(
            "index.html", dietary_options=DIETARY_OPTIONS, mode=mode,
            error="Choose a food photo first.",
        ), 400
    if mode == "ingredients" and not ingredients:
        return render_template(
            "index.html", dietary_options=DIETARY_OPTIONS, mode=mode,
            error="List the ingredients you have, or choose Help me decide.",
        ), 400

    restrictions = [
        DIETARY_OPTIONS[key]
        for key in request.form.getlist("dietary")
        if key in DIETARY_OPTIONS
    ]
    custom_restrictions = request.form.get("custom_restrictions", "").strip()[:500]
    try:
        max_calories = int(request.form.get("max_calories", "") or 0)
        if max_calories < 0 or max_calories > 3000:
            max_calories = 0
    except ValueError:
        max_calories = 0

    try:
        client = sarvam_client()
        image_summary = None
        if mode == "discover":
            flavors = request.form.getlist("flavor")
            textures = request.form.getlist("texture")
            meal = request.form.get("meal_type", "any meal")
            cuisine = request.form.get("cuisine", "any Indian cuisine")
            time_limit = request.form.get("time_limit", "any cooking time")
            preferences = [
                f"Flavor preferences: {', '.join(flavors) if flavors else 'surprise me'}. ",
                f"Texture preferences: {', '.join(textures) if textures else 'any texture'}. ",
                f"Meal: {meal}. Cuisine: {cuisine}. Cooking time: {time_limit}.",
            ]
            source_text = "Help me decide what to eat. I have no ingredients selected. " + "".join(preferences)
        elif mode == "photo":
            if image is None:
                image_summary = request.form.get("photo_context", "").strip()[:1200]
            else:
                image_bytes = image.read(MAX_UPLOAD_BYTES + 1)
                if len(image_bytes) > MAX_UPLOAD_BYTES:
                    return render_template(
                        "index.html", dietary_options=DIETARY_OPTIONS,
                        error="Choose an image smaller than 8 MB.",
                    ), 413
                mime_type = (image.mimetype or "").lower()
                if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
                    return render_template(
                        "index.html", dietary_options=DIETARY_OPTIONS,
                        error="Upload a JPG, PNG, or WebP food photo.",
                    ), 400
                image_summary = identify_food(client, image_bytes, mime_type)
            source_text = "Use this food photo analysis as the starting point for recipe ideas: " + image_summary
        elif mode == "ingredients":
            source_text = "Ingredients currently available: " + ingredients[:2000]
        else:
            return render_template(
                "index.html", dietary_options=DIETARY_OPTIONS, mode=mode,
                error="Choose photo, ingredients, or Help me decide.",
            ), 400

        if request.form.get("refresh") == "1":
            previous_names = request.form.get("previous_recipe_names", "").strip()[:500]
            if previous_names:
                source_text += (
                    " Generate fresh alternatives and avoid repeating these previous recipe names: "
                    + previous_names + "."
                )

        result = build_recipe_request(
            client, source_text, restrictions, custom_restrictions, max_calories,
            discovery_mode=(mode == "discover"),
        )
        recipes = result.get("recipes", [])
        if not isinstance(recipes, list):
            recipes = []
        return render_template(
            "index.html", dietary_options=DIETARY_OPTIONS, recipes=recipes,
            image_summary=image_summary, source_text=source_text,
            selected_dietary=request.form.getlist("dietary"),
            mode=mode,
            selected_flavors=request.form.getlist("flavor"),
            selected_textures=request.form.getlist("texture"),
            meal_type=request.form.get("meal_type", "any"),
            cuisine=request.form.get("cuisine", "any"),
            time_limit=request.form.get("time_limit", "any"),
            custom_restrictions=custom_restrictions,
            max_calories=request.form.get("max_calories", ""),
            ingredients=ingredients,
        )
    except RuntimeError as error:
        message = str(error)
    except Exception as error:
        app.logger.exception("Recipe recommendation request failed")
        message = api_error_message(error)
    return render_template(
        "index.html", dietary_options=DIETARY_OPTIONS,
        selected_dietary=request.form.getlist("dietary"),
        custom_restrictions=custom_restrictions,
        max_calories=request.form.get("max_calories", ""),
        ingredients=ingredients, error=message,
    ), 502


@app.post("/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify(error="Record or select an audio clip first."), 400
    audio_bytes = audio.read(MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        return jsonify(error="Choose an audio clip smaller than 8 MB."), 413
    language_code = request.form.get("language_code", "en-IN")
    if language_code not in DICTATION_LANGUAGES:
        return jsonify(error="Choose a supported dictation language."), 400
    mime_type = (audio.mimetype or "").split(";", 1)[0].strip().lower()
    codec_by_mime = {
        "audio/webm": "webm",
        "audio/mp4": "mp4",
        "audio/x-m4a": "x-m4a",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "x-wav",
        "audio/aac": "aac",
        "audio/flac": "flac",
    }
    input_audio_codec = codec_by_mime.get(mime_type)
    if not input_audio_codec:
        return jsonify(error=f"This browser recording format is not supported: {mime_type or 'unknown'}"), 415
    try:
        response = sarvam_client().speech_to_text.transcribe(
            file=(audio.filename, io.BytesIO(audio_bytes), mime_type),
            model="saaras:v4",
            mode="transcribe",
            language_code=language_code,
            input_audio_codec=input_audio_codec,
        )
        transcript = getattr(response, "transcript", "")
        if not transcript and hasattr(response, "model_dump"):
            transcript = response.model_dump().get("transcript", "")
        return jsonify(transcript=transcript)
    except RuntimeError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Speech transcription request failed")
        return jsonify(error=api_error_message(error)), 502


@app.post("/speak")
def speak():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify(error="There is no recipe text to read."), 400
    language_code = request.form.get("language_code", "en-IN")
    voice = TTS_LANGUAGES.get(language_code)
    if not voice:
        return jsonify(error="Choose a supported read-aloud language."), 400
    text = text[:2000]
    try:
        client = sarvam_client()
        if language_code != "en-IN":
            translation = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code=language_code,
                model="mayura:v1",
                mode="modern-colloquial",
            )
            text = (translation.translated_text or "").strip()[:2400]
            if not text:
                return jsonify(error="Sarvam could not translate this recipe."), 502
        speech = client.text_to_speech.convert(
            text=text,
            model="bulbul:v4-flash",
            language_code=language_code,
            speaker=voice["speaker"],
            output_audio_codec="mp3",
        )
        audio_bytes = base64.b64decode("".join(speech.audios or []))
        if not audio_bytes:
            return jsonify(error="Sarvam did not return audio."), 502
        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", download_name="recipe.mp3")
    except RuntimeError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Text-to-speech request failed")
        return jsonify(error=api_error_message(error)), 502


@app.errorhandler(413)
def request_too_large(_error):
    return render_template(
        "index.html", dietary_options=DIETARY_OPTIONS,
        error="The upload is too large. Keep photos and audio clips under 8 MB.",
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
