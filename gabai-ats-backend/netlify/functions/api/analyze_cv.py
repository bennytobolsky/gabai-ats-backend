import os
import json
import traceback
import base64
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

def extract_clean_json(text):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_json_str = text[start_idx:end_idx+1]
            return json.loads(clean_json_str)
        return None
    except:
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        body = request.get_json() or {}
        if isinstance(body, str):
            try: body = json.loads(body)
            except: pass
                
        if not isinstance(body, dict):
            return jsonify({"error": "Invalid request body format."}), 400

        cv_base64 = body.get("cv_base64", "")
        job_context = body.get("job_context", "לא צוין") 
        
        if not cv_base64:
            return jsonify({"error": "No cv_base64 provided in the request"}), 400

        try:
            pdf_bytes = base64.b64decode(cv_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400

        cv_text = ""
        image_messages = []
        
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                # 1. ניסיון לחלץ טקסט רגיל
                for page in doc:
                    cv_text += page.get_text()
                
                # 2. נשק יום הדין (Vision): אם הטקסט ריק כמעט לגמרי, מצלמים את הקובץ
                if len(cv_text.strip()) < 50:
                    cv_text = "הטקסט לא חולץ כראוי בגלל קידוד הפונט, לכן מצורפות תמונות של המסמך לקריאה ויזואלית:"
                    # ניקח מקסימום את 3 העמודים הראשונים כדי למקד את ה-AI
                    for page in doc[:3]:
                        pix = page.get_pixmap(dpi=150) # רזולוציה אופטימלית לקריאה
                        img_base64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
                        image_messages.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        })
        except Exception as pdf_error:
            return jsonify({"error": f"Failed to process PDF: {str(pdf_error)}"}), 500

        # אתחול הלקוח של OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        system_prompt = (
            "אתה עוזר גיוס בכיר ומקצועי בסוכנות ביטוח ופנסיה מובילה בישראל. "
            "תפקידך לנתח קורות חיים אל מול דרישות משרה ספציפית, באובייקטיביות מלאה. "
            "עליך להחזיר תמיד אך ורק פלט במבנה JSON תקין בשפה העברית, ללא שום טקסט נוסף לפני או אחרי ה-JSON."
        )
        
        job_context_text = json.dumps(job_context, ensure_ascii=False) if isinstance(job_context, dict) else str(job_context)
            
        user_prompt_text = f"""
        דרישות המשרה:
        {job_context_text}

        קורות החיים של המועמד:
        ---
        {cv_text}
        ---

        עליך לנתח את קורות החיים (קרא את הטקסט או את התמונות המצורפות של המסמך) ולהחזיר JSON במבנה הבא בדיוק.
        חשוב מאוד: חלץ את המידע האמיתי מתוך המסמך.
        {{
          "full_name": "השם המלא האמיתי של המועמד",
          "phone": "מספר הטלפון האמיתי של המועמד",
          "email": "כתובת האימייל האמיתית של המועמד",
          "id_number": "מספר תעודת הזהות של המועמד. אם לא רשום במסמך, החזר null",
          "match_score": מספר בלבד בין 0 ל-100 על סמך מידת ההתאמה לדרישות,
          "ai_feedback": "3-5 משפטים בעברית המסכמים את ההתרשמות הכללית בצורה ישירה ועניינית",
          "ai_strengths": "נקודות חוזקה מרכזיות של המועמד שמתאימות במדויק לדרישות",
          "ai_gaps": "פערים, חוסר ניסיון, או דרישות חובה שחסרות",
          "ai_red_flags": "נורות אזהרה בולטות במידה ויש. אם אין, רשום null"
        }}
        """

        # בניית חבילת הנתונים ל-AI (טקסט + תמונות אם יש)
        content_array = [{"type": "text", "text": user_prompt_text}]
        if image_messages:
            content_array.extend(image_messages)

        # פנייה ל-API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_array}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        raw_content = response.choices[0].message.content
        result_json = extract_clean_json(raw_content)
        
        if result_json is None:
            return jsonify({"error": "Failed to parse AI response into a valid JSON object", "raw_response": raw_content}), 500
            
        return jsonify(result_json), 200

    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({"error": str(e), "details": error_details}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running with OCR Vision fallback!"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
