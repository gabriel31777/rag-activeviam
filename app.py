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
    
    # Strict command requested:
    command = [
        sys.executable, 
        "src/09_rag_agent_groq.py",
        "--q", question,
        "--mode", "chat",
        "--answer-style", style
    ]
    
    # Show the executed command in the console
    print("\n" + "="*50)
    print(f"🚀 EXÉCUTION DE LA COMMANDE :\n{' '.join(command)}")
    print("="*50 + "\n")

    try:
        # Execute the script. 
        # IMPORTANT: we removed 'text=True' and 'encoding' to receive raw bytes.
        # This avoids crashes with accented characters in paths.
        result = subprocess.run(
            command, 
            capture_output=True, 
            cwd=os.getcwd()
        )
        
        # Decode manually ignoring accent errors
        # errors='replace' will swap problematic characters with a symbol without crashing.
        answer = result.stdout.decode('utf-8', errors='replace').strip()
        error_output = result.stderr.decode('utf-8', errors='replace').strip()
        
        # If the answer is empty, check stderr for errors
        if result.returncode != 0 or not answer:
            if error_output:
                print(f"❌ ERREUR (Logs) :\n{error_output}")
                # Sometimes the program runs but emits warnings in stderr, so we only show an error if answer is empty
                if not answer:
                    answer = f"Erreur lors de l'exécution : {error_output}"
            
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'answer': str(e)})

if __name__ == '__main__':
    print("✅ Serveur en ligne ! Ouvrez http://127.0.0.1:5000")
    app.run(debug=True, port=5000)