from pipeline import Pipeline
import glob

def demo():
    p = Pipeline(use_ollama_extractor=False)

    for path in glob.glob("transcripts/Командос/*.txt"):
        name = path.split("/")[-1].rsplit(".",1)[0]
        parts = name.split("_",1)
        if len(parts)==2:
            meeting_type, meeting_id = parts[0], parts[1]
        else:
            meeting_id = name
            meeting_type = None
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        p.ingest_meeting(meeting_id=meeting_id, transcript=txt, meeting_type=meeting_type, metadata={"source_file": path})
        print("Indexed", meeting_id, "type", meeting_type)

    q = "Кто назначен ответственным за релиз?"
    out = p.answer_query(q)
    print("Answer:", out["answer"])
    print("Provenance:", out["provenance"])

if __name__ == "__main__":
    demo()
