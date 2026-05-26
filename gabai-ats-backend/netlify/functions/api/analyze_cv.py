import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

# יצירת שרת האינטרנט
app = Flask(__name__)

# הגדרת נתיב הקבלה של השרת
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. קבלת הנתונים שנשלחו מהאוטומציה של Base44
        body = request.get_json() or {}
        cv_text = body.get("cv_text", "")
        job_context = body.get("job_context", {})
        
        if not cv_text:
            return jsonify({"error": "No CV text provided"}), 400

        # 2. אתחול הלקוח של OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # 3. בניית הפרומפט לסוכן ה-AI
        system_prompt = (
            "אתה עוזר גיוס בכיר ומקצועי בסוכנות ביטוח ופנסיה מובילה בישראל. "
            "תפקידך לנתח קורות חיים אל מול דרישות משרה ספציפית, באובייקטיביות מלאה. "
            "עליך להחזיר תמיד אך ורק פלט במבנה JSON תקין בשפה העברית, ללא שום טקסט נוסף לפני או אחרי ה-JSON."
        )
        
        user_prompt = f"""
        דרישות המשרה:
        - שם המשרה: {job_context.get('title', 'לא צוין')}
        - מילות מפתח קריטיות: {job_context.get('keywords', 'לא צוין')}
        - ניסיון מינימלי נדרש: {job_context.get('min_experience_years', '0')} שנים
        - רישיונות נדרשים: {job_context.get('required_licenses', 'ללא')}
        - תיאור המשרה המלא: {job_context.get('requirements_text', 'לא צוין')}

        קורות החיים של המועמד (טקסט גולמי):
        ---
        {cv_text}
        ---

        עליך לנתח את קורות החיים ולהחזיר JSON במבנה הבא בדיוק (הערכים במחרוזות צריכים להיות בעברית):
        {{
          "full_name": "שם המועמד המלא כפי שמופיע בקורות החיים",
          "phone": "מספר הטלפון של המועמד",
          "email": "כתובת האימייל של המועמד",
          "match_score": מספר בלבד בין 0 ל-100 על סמך מידת ההתאמה לדרישות,
          "ai_feedback": "3-5 משפטים בעברית המסכמים את ההתרשמות הכללית בצורה ישירה ועניינית",
          "ai_strengths": "נקודות חוזקה מרכזיות של המועמד שמתאימות במדויק לדרישות המשרה",
          "ai_gaps": "פערים, חוסר ניסיון, או דרישות חובה של המשרה שלא קיימות בקורות החיים",
          "ai_red_flags": "נורות אזהרה בולטות במידה ויש (כמו: אי-יציבות תעסוקתית חריגה, מעברי עבודות תכופים). אם אין, רשום null"
        }}
        """

        # 4. פנייה ל-API של OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        # 5. חילוץ התשובה והחזרתה
        result_json = json.loads(response.choices[0].message.content)
        return jsonify(result_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# נתיב בדיקה כדי לוודא שהשרת באוויר
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Server is running perfectly!"}), 200

if __name__ == '__main__':
    # הפעלת השרת
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
