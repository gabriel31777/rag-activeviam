from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os

# Flask Configuration
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    style = data.get('style')  # 'value' ou 'free'
    embedding = data.get('embedding', 'hybrid')

    command = [
        sys.executable,
        "src/04_rag_agent.py",
        "--q", question,
        "--mode", "chat",
        "--answer-style", style,
        "--embedding", embedding
    ]

    print("\n" + "="*50)
    print(f"🚀 EXÉCUTION DE LA COMMANDE :\n{' '.join(command)}")
    print("="*50 + "\n")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            cwd=os.getcwd()
        )

        answer = result.stdout.decode('utf-8', errors='replace').strip()
        error_output = result.stderr.decode('utf-8', errors='replace').strip()

        if result.returncode != 0 or not answer:
            if error_output:
                print(f"❌ ERREUR (Logs) :\n{error_output}")
                if not answer:
                    answer = f"Erreur lors de l'exécution : {error_output}"

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': str(e)})

if __name__ == '__main__':
    print("✅ Serveur en ligne ! Ouvrez http://127.0.0.1:5000")
    app.run(debug=True, port=5000)