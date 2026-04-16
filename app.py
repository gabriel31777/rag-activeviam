import os
os.environ["PYTHONUNBUFFERED"] = "1"

from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import threading

# Config Flask
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    style = data.get('style')
    embedding = data.get('embedding', 'hybrid')

    command = [
        sys.executable, "-u",
        "src/04_rag_agent.py",
        "--q", question,
        "--mode", "chat",
        "--answer-style", style,
        "--embedding", embedding
    ]

    SKIP = ('UserWarning', 'warnings.warn', 'FutureWarning', 'site-packages', 'NOTE: Redirects')

    sys.stderr.write("\n" + "=" * 50 + "\n")
    sys.stderr.write(f"  QUESTION: {question}\n")
    sys.stderr.write(f"  EMBEDDING: {embedding} | STYLE: {style}\n")
    sys.stderr.write("=" * 50 + "\n")

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            env=env
        )

        # Thread de lecture stderr en temps reel
        def stream_logs():
            for raw_line in iter(process.stderr.readline, b''):
                line = raw_line.decode('utf-8', errors='replace').rstrip()
                if not line or any(p in line for p in SKIP):
                    continue
                sys.stderr.write(f"  {line}\n")
                sys.stderr.flush()

        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()


        process.wait()
        log_thread.join(timeout=2)

        answer = process.stdout.read().decode('utf-8', errors='replace').strip()
        process.stdout.close()

        if process.returncode != 0 and not answer:
            answer = "Erreur lors du traitement de votre question. Veuillez reessayer."

        preview = answer[:120] + ('...' if len(answer) > 120 else '')
        sys.stderr.write(f"\n  [REPONSE] {preview}\n")
        sys.stderr.write("=" * 50 + "\n\n")
        sys.stderr.flush()

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': str(e)})

if __name__ == '__main__':
    print("[OK] Serveur en ligne ! Ouvrez http://127.0.0.1:5000")
    app.run(debug=True, port=5000)