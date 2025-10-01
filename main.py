# main.py
import argparse, glob, os, logging
from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

def demo_index_and_query():
    p = Pipeline()
    for path in glob.glob("transcripts/Командос/*.txt"):
        name = os.path.basename(path).rsplit(".", 1)[0]
        meeting_type, meeting_id = (name.split("_", 1) + [None])[:2] if "_" in name else (None, name)
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        p.ingest_meeting(meeting_id, txt, meeting_type, {"source_file": path})
        logger.info(f"Indexed {meeting_id}, type={meeting_type}")
    q = "Кто назначен ответственным за релиз?"
    out = p.answer_query(q)
    logger.info(f"Answer: {out['answer']}")
    logger.info(f"Provenance: {out['provenance']}")

def build_profile_cli(name, files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    expanded_files = []
    for f in files:
        expanded_files.extend(glob.glob(f))
    if not expanded_files:
        raise ValueError(f"No meeting files found for patterns: {files}")
    p = Pipeline()
    res = p.build_profile_from_files(name, expanded_files, output_dir=output_dir)
    logger.info(f"Profile saved: {res['txt_path']}, {res['docx_path']}")
    logger.info(f"Profile saved in KAG id/version: {res['saved_profile']}")

def main():
    parser = argparse.ArgumentParser(description="Meeting pipeline CLI")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--build-profile")
    parser.add_argument("--meeting-files", nargs='*')
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.demo:
        demo_index_and_query()
    elif args.build_profile and args.meeting_files:
        build_profile_cli(args.build_profile, args.meeting_files, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
