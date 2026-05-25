import os
import json
from openai import OpenAI

def handler(event, context):
    # תמיכה בבקשות מסוג POST בלבד
    if event.get("httpMethod") != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method Not Allowed"})
        }

    try:
        # 1. קבלת הנתונים שנשלחו מהאוטומציה של Base44
        body = json.loads(event.get("body", "{}"))
        cv_text = body.get("cv_text", "")
        job_context = body.get("job_context", {})
        
        if not cv_text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No CV text provided"})
            }

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
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result_json, ensure_ascii=False)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
