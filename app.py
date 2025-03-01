from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from skills_recommendation import get_skills_recommendation

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_skills_recommendation', methods=['POST', 'GET'])
def get_skills_recommendation_route():
    try:
        contents = request.get_json()
        job_occupation = contents['job_occupation']
        resume_contents = contents['resume_contents']
        
        result, context = get_skills_recommendation(job_occupation, resume_contents)
        return jsonify({"skills": result, "context": context})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)})