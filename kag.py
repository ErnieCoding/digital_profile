# kag.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import KAG_DB
import difflib

Base = declarative_base()
engine = create_engine(f"sqlite:///{KAG_DB}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

class Meeting(Base):
    __tablename__ = "meetings"
    id = Column(String, primary_key=True)
    meeting_type = Column(String, index=True)
    title = Column(String, nullable=True)
    date = Column(String, nullable=True)
    meta = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Participant(Base):
    __tablename__ = "participants"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    meta = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String, index=True, nullable=True)
    subject = Column(String, index=True)
    predicate = Column(String)
    object = Column(Text)
    confidence = Column(Float, default=0.5)
    source = Column(String)
    meta = Column("metadata", JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    content = Column(Text)
    version = Column(Integer, default=1)
    source_meetings = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

class KnowledgeStore:
    def __init__(self):
        self.s = Session()

    def upsert_meeting(self, meeting_id: str, meeting_type: str | None = None, metadata: dict | None = None):
        m = self.s.query(Meeting).filter_by(id=meeting_id).first()
        if m:
            if meeting_type:
                m.meeting_type = meeting_type
            if metadata:
                current = (m.meta or {})
                current.update(metadata)
                m.meta = current
            self.s.commit()
            return m
        m = Meeting(id=meeting_id, meeting_type=meeting_type, meta=metadata or {})
        self.s.add(m)
        self.s.commit()
        return m

    def upsert_participant(self, name: str, metadata: dict | None = None):
        p = self.s.query(Participant).filter_by(name=name).first()
        if p:
            if metadata:
                current = (p.meta or {})
                current.update(metadata)
                p.meta = current
            self.s.commit()
            return p
        p = Participant(name=name, meta=metadata or {})
        self.s.add(p)
        self.s.commit()
        return p

    def add_fact(self, meeting_id: str | None, subject: str, predicate: str, obj: str, confidence: float = 0.6, source: str | None = None, metadata: dict | None = None):
        existing = self.s.query(Fact).filter(Fact.subject==subject).all()
        for e in existing:
            s = f"{e.subject} {e.predicate} {e.object}"
            t = f"{subject} {predicate} {obj}"
            ratio = difflib.SequenceMatcher(None, s, t).ratio()
            if ratio > 0.86:
                e.confidence = max(e.confidence, confidence)
                if metadata:
                    mm = (e.meta or {})
                    mm.update(metadata)
                    e.meta = mm
                self.s.commit()
                return e
        f = Fact(meeting_id=meeting_id, subject=subject, predicate=predicate, object=obj, confidence=confidence, source=source, meta=metadata or {})
        self.s.add(f)
        self.s.commit()
        return f

    def query_facts(self, keywords: list[str] | None = None, subject: str | None = None, limit: int = 50):
        q = self.s.query(Fact).filter(Fact.is_active==True)
        if subject:
            q = q.filter(Fact.subject.ilike(f"%{subject}%"))
        if keywords:
            for kw in keywords:
                q = q.filter((Fact.subject.ilike(f"%{kw}%")) | (Fact.predicate.ilike(f"%{kw}%")) | (Fact.object.ilike(f"%{kw}%")))
        return q.order_by(Fact.created_at.desc()).limit(limit).all()

    def get_meeting_type_candidates_for_query(self, query: str) -> list[str]:
        kws = [w for w in query.split() if len(w) > 3]
        counts = {}
        for kw in kws:
            facts = self.s.query(Fact).filter((Fact.subject.ilike(f"%{kw}%")) | (Fact.predicate.ilike(f"%{kw}%")) | (Fact.object.ilike(f"%{kw}%"))).all()
            for f in facts:
                if f.meeting_id:
                    m = self.s.query(Meeting).filter_by(id=f.meeting_id).first()
                    if m and m.meeting_type:
                        counts[m.meeting_type] = counts.get(m.meeting_type, 0) + 1
        if not counts:
            rows = self.s.query(Meeting.meeting_type).all()
            for r in rows:
                if r[0]:
                    counts[r[0]] = counts.get(r[0], 0) + 1
        sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [t for t,c in sorted_types]

    # --- new profile methods ---
    def save_profile(self, name: str, content: str, source_meetings: list | None = None):
        last = self.s.query(Profile).filter(Profile.name==name).order_by(Profile.version.desc()).first()
        version = (last.version + 1) if last else 1
        p = Profile(name=name, content=content, version=version, source_meetings=source_meetings or [])
        self.s.add(p)
        self.s.commit()
        return p

    def get_latest_profile(self, name: str):
        return self.s.query(Profile).filter(Profile.name==name).order_by(Profile.version.desc()).first()

    def list_profiles(self, name: str):
        return self.s.query(Profile).filter(Profile.name==name).order_by(Profile.version.desc()).all()
