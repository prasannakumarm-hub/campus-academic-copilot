# Campus Academic Copilot — Complete Hackathon MVP

A runnable web application using Streamlit, OpenAI embeddings and chat completions, local NumPy cosine-similarity retrieval, multi-document comparison, and timetable clash detection.

## 1. Requirements
- Python 3.10 or newer
- OpenAI API key with access to a chat model and `text-embedding-3-small`
- Internet access for OpenAI API requests

## 2. Install and run (Windows PowerShell)
```powershell
cd campus_academic_copilot_complete
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and paste your API key
streamlit run app.py
```

## 3. Install and run (macOS/Linux)
```bash
cd campus_academic_copilot_complete
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your API key
streamlit run app.py
```
You can also paste the key in the app sidebar. Never share or commit API keys.

## 4. First-run demo
1. Launch app.
2. In the sidebar click **Load included demo documents**.
3. Wait for embeddings to be created. The app stores the index locally in `.copilot_store/`.
4. Ask: `What attendance requirement is stated in the sample regulation?`
5. Ask: `List the experiments in the DAA laboratory manual.`
6. Ask: `What course-file evidence is mentioned in the NBA checklist?`
7. In Compare documents, compare `demo_regulations.txt` and `demo_student_guidelines.txt`.
8. In Timetable intelligence, upload `data/demo_timetable.csv`. It intentionally contains a faculty overlap and a room overlap.

## 5. Supported file formats
PDF, DOCX, TXT and CSV. PDF page numbers are retained. DOCX/TXT/CSV are treated as page 1. Scanned PDFs need OCR; OCR is not included in this overnight build.

## 6. Team allocation (5+)
- Member 1: UI, Streamlit layout, demo flow.
- Member 2: document extraction, sample files, test ingestion.
- Member 3: embeddings, semantic retrieval, prompt grounding.
- Member 4: timetable CSV validation and conflict logic.
- Member 5: integration, QA, citation verification.
- Member 6: pitch, impact narrative, judge Q&A.

## 7. 5-minute judging demo
- 0:00–0:40 Problem and solution.
- 0:40–2:10 Upload/load documents and ask a cross-document question.
- 2:10–3:00 Show evidence references and explain abstention when evidence is absent.
- 3:00–4:00 Compare regulations and guidelines.
- 4:00–4:40 Detect timetable clashes.
- 4:40–5:00 Impact and roadmap.

## 8. Important limitations / honest claims
- This is a complete runnable hackathon prototype, not production-ready institutional software.
- The index uses OpenAI embeddings with local cosine similarity; it is a small-scale local index, not a managed vector database.
- Role selection is for demonstration only; it does not enforce security. Add institutional SSO and server-side authorization before real deployment.
- No OCR for image-only/scanned PDFs.
- Timetable checker requires exact columns: `faculty,day,start,end,room,course`; times must be 24-hour `HH:MM`.
- Demo documents contain fictional content. They are not official VTU, NBA, NAAC or institutional rules.
- Do not send confidential student records or restricted institutional data to external APIs without explicit institutional approval.
- Human verification is required for academic, regulatory and accreditation decisions.

## 9. Troubleshooting
- `Incorrect API key`: verify `.env` value and restart Streamlit.
- `model not found`: change chat model in sidebar to a model enabled for your API account.
- Embedding permission error: ensure the API project has access to the embedding model.
- Slow first run: initial embedding requests take time; subsequent reruns use the local index unless files change.
- Empty answer: rephrase question or add the relevant document; the app deliberately avoids guessing.

## 10. Project structure
```
campus_academic_copilot_complete/
  app.py
  requirements.txt
  .env.example
  README.md
  data/
    demo_regulations.txt
    demo_student_guidelines.txt
    demo_daa_lab_manual.txt
    demo_nba_checklist.txt
    demo_syllabus.pdf
    demo_timetable.csv
  tests/
    test_timetable.py
```
