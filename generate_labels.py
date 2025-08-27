# ---- Label generator (LLM optional, with robust fallback) ----
from typing import List, Optional
import json, re, os
class GenerateLabels():
    def __init__(self, use_llm: bool = False,
                 max_labels: int = 10,
                 include_synonyms: bool = False,
                 nested: bool = True):
        self.labels = self.make_office_workshop_labels(use_llm=True, 
                                                       max_labels=10, 
                                                       include_synonyms=False,
                                                         nested=True)


    def get_labels(self) -> List[List[str]]:
        return self.labels
    def _default_office_workshop_taxonomy(self):
        base = [
            # people & body parts
           # "person","face","head","hand","arm","leg",
            # furniture & room
            "desk","office chair","table","shelf","cabinet","workbench","whiteboard","window","door","floor","ceiling","wall",
            # computers & peripherals
            "laptop","computer monitor","keyboard","computer mouse","webcam","phone","tablet","router","printer","server rack",
            # stationery & desk items
            #"notebook","pen","pencil","sticky notes","backpack","coffee cup","mug","water bottle","trash can","power outlet","cable",
            # workshop/maker
            #"toolbox","screwdriver","drill","hammer","wrench","pliers","tape measure","3D printer","filament spool","safety glasses","gloves",
            # misc
            #"book","binder","box","lamp","plant","clock"
        ]
        parts = {
            #"person": ["face","torso","hand","arm","leg"],
            #"face": ["nose","eye","mouth","ear"],
            "laptop": ["screen","keyboard","touchpad","webcam","logo","port"],
            #"computer monitor": ["screen","bezel","stand","power button"],
            "office chair": ["backrest","seat","armrest","wheels"],
            "desk": ["drawer","leg","cable grommet"],
            #"window": ["frame","glass","handle","blinds"],
            #"door": ["handle","hinge","lock"],
            "keyboard": ["keycap","spacebar","enter key"],
            #"mouse": ["scroll wheel","cable"],
            #"phone": ["screen","camera","charging port"],
            #"backpack": ["strap","zipper","pocket"],
            #"coffee cup": ["lid","handle"],
            #"water bottle": ["cap","label"],
            #"workbench": ["vise","drawer"],
            #"toolbox": ["drawer","handle","latch"],
            #"3D printer": ["nozzle","bed","gantry","filament"],
            #"server rack": ["switch","cable","PDU"],
            #"printer": ["paper tray","control panel","output slot"]
        }
        synonyms = {
            "computer monitor": ["monitor","display","screen"],
            "phone": ["smartphone","mobile phone"],
            "office chair": ["swivel chair","task chair"],
            "trash can": ["bin","garbage can"],
            "router": ["wifi router","wireless router","network switch"],
            "3D printer": ["FDM printer"]
        }
        return base, parts, synonyms

    def _expand_hierarchical(self, base, parts, synonyms,
                            max_labels=10, include_synonyms=True, nested=True,
                            ensure_nested_min=24):
        # Build tiers
        flat = list(base)
        if include_synonyms:
            for k, vs in synonyms.items():
                flat.extend(vs)

        obj_part = []
        obj_part_sub = []
        if nested:
            for obj, ps in parts.items():
                for p in ps:
                    obj_part.append(f"{obj}[{p}]")
                    if p in parts:
                        for sub in parts[p]:
                            obj_part_sub.append(f"{obj}[{p}[{sub}]]")

        # Also allow standalone parts directly
        parts_only = sorted({p for lst in parts.values() for p in lst})

        # Dedup while preserving order
        def dedup(seq):
            seen = set(); out = []
            for s in seq:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        obj_part = dedup(obj_part)
        obj_part_sub = dedup(obj_part_sub)
        flat = dedup(flat)
        parts_only = dedup(parts_only)

        out = []
        # 1) take nested first up to ensure_nested_min (split across both tiers)
        half = ensure_nested_min // 2
        out.extend(obj_part[:half])
        out.extend(obj_part_sub[:ensure_nested_min - len(out)])

        # 2) then remaining nested
        out.extend(obj_part[half:])
        out.extend(obj_part_sub)

        # 3) then parts-only, then flat
        out.extend(parts_only)
        out.extend(flat)

        # Final sort to keep more nested toward the front, then trim
        out = dedup(out)
        out = sorted(out, key=lambda s: (-s.count('['), s))[:max_labels]
        return [out]


    def propose_labels_with_llm(self, description: str = "indoor office and workshop",
                                seed_labels: Optional[List[str]] = None,
                                max_labels: int = 10) -> Optional[List[List[str]]]:
        """
        Uses OpenAI if OPENAI_API_KEY is set; returns [[...]] or None on failure.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            sys = "You generate compact visual labels/parts for object detection prompts. Use bracket notation object[part[subpart]] when meaningful. Return JSON with a 'labels' array. Keep <= {} items, prioritize office/workshop.".format(max_labels)
            user = f"Subject: {description}. Examples: person[face[nose]], laptop[webcam], window, desk, office chair."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}]
            )
            txt = resp.choices[0].message.content
            m = re.search(r"\{.*\}", txt, re.S)
            data = json.loads(m.group(0)) if m else {"labels": []}
            labels = [s.strip() for s in data.get("labels", []) if s.strip()]
            if not labels:
                return None
            return [[*labels][:max_labels]]
        except Exception:
            return None

    def make_office_workshop_labels(self, use_llm: bool = False,
                                    max_labels: int = 10,
                                    include_synonyms: bool = True,
                                    nested: bool = True) -> List[List[str]]:
        if use_llm:
            llm_labels = self.propose_labels_with_llm(max_labels=max_labels)
            if llm_labels:
                return llm_labels
        base, parts, syn = self._default_office_workshop_taxonomy()
        return self._expand_hierarchical(base, parts, syn,
                                    max_labels=max_labels,
                                    include_synonyms=include_synonyms,
                                    nested=nested)
    # ---- end label generator ----


if __name__ == "__main__":
    gl = GenerateLabels(use_llm=True, max_labels=10, include_synonyms=True, nested=True)
    labels = gl.get_labels()
    #print("Final labels:", labels[0])
    print("labels",labels)