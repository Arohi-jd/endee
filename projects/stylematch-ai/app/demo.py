from __future__ import annotations

from app.bootstrap_index import main as bootstrap_main
from app.recommend import main as recommend_main


if __name__ == "__main__":
    # Convenience script:
    # 1) index the sample catalog into Endee
    # 2) then call recommend.py with CLI args if needed
    # Prefer running bootstrap_index.py and recommend.py directly for full control.
    bootstrap_main()
    recommend_main()
