import os
import json
import traceback
from flask import Flask, request, jsonify
from openai import OpenAI

# יצירת שרת האינטרנט
app = Flask(__name__)

# פונקציית עזר לניקוי וחילוץ JSON מתשובת ה-AI
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

# הגדרת נתיב הקבלה של השרת
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. קבלת הנתונים שנשלחו מהאוטומציה של Make
        body = request.get_json() or {}
        
        # הגנה למקרה שהנתונים נשלחו בפורמט לא צפוי
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except:
                pass
                
        if not isinstance(body, dict):
            return jsonify({"error": "Invalid request body format."}), 400

        cv_text = body.get("cv_text", "")
        # אנחנו פשוט לוקחים את כל הבלוק כטקסט
        job_context = body.get("job_context", "לא צוין") 
        
        if not cv_text:
            return jsonify({"error": "No CV text provided"}), 400

        # 2. אתחול הלקוח של OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        system_prompt = (
            "אתה עוזר גיוס בכיר ומקצועי בסוכנות ביטוח ופנסיה מובילה בישראל. "
            "תפקידך לנתח קורות חיים אל מול דרישות משרה ספציפית, באובייקטיביות מלאה. "
            "עליך להחזיר תמיד אך ורק פלט במבנה JSON תקין בשפה העברית, ללא שום טקסט נוסף לפני או אחרי ה-JSON."
        )
        
        # המרת דרישות המשרה לטקסט רגיל כדי למנוע שגיאות
        job_context_text = json.dumps(job_context, ensure_ascii=False) if isinstance(job_context, dict) else str(job_context)
            
        user_prompt = f"""
        דרישות המשרה:
        {job_context_text}

        קורות החיים של המועמד (טקסט גולמי):
        ---
        {cv_text}
        ---

        עליך לנתח את קורות החיים המוצגים למעלה ולהחזיר JSON במבנה הבא בדיוק.
        חשוב מאוד: עליך לחלץ את המידע האמיתי מתוך קורות החיים של המועמד (לדוגמה, תחת full_name רשום את השם האמיתי שמופיע בקובץ):
        {{
          "full_name": "השם המלא האמיתי של המועמד מתוך קורות החיים",
          "phone": "מספר הטלפון האמיתי של המועמד מתוך קורות החיים",
          "email": "כתובת האימייל האמיתית של המועמד מתוך קורות החיים",
          "id_number": "מספר תעודת הזהות של המועמד (חפש כל צירוף אפשרי כמו מספר זהות או ת.ז). אם לא רשום במסמך, החזר null",
          "match_score": מספר בלבד בין 0 ל-100 על סמך מידת ההתאמה לדרישות,
          "ai_feedback": "3-5 משפטים בעברית המסכמים את ההתרשמות הכללית בצורה ישירה ועניינית",
          "ai_strengths": "נקודות חוזקה מרכזיות של המועמד שמתאימות במדויק לדרישות",
          "ai_gaps": "פערים, חוסר ניסיון, או דרישות חובה שחסרות",
          "ai_red_flags": "נורות אזהרה בולטות במידה ויש. אם אין, רשום null"
        }}
        """

        # 4. פנייה ל-API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        # 5. חילוץ וניקוי
        raw_content = response.choices[0].message.content
        result_json = extract_clean_json(raw_content)
        
        if result_json is None:
            return jsonify({"error": "Failed to parse AI response into a valid JSON object", "raw_response": raw_content}), 500
            
        return jsonify(result_json), 200

    except Exception as e:
        # במקרה של שגיאה חדשה, נדפיס בדיוק באיזו שורה היא קרתה
        error_details = traceback.format_exc()
        return jsonify({"error": str(e), "details": error_details}), 500

# נתיב בדיקה כדי לוודא שהשרת באוויר
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running perfectly!"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
