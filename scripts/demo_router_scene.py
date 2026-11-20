"""Create a complete router explainer plan for the all-in-one notebook.
Run: python -m scripts.demo_router_scene --output /path/scene.json
"""
import argparse
from pathlib import Path
from voice2reel.production_contract import ProductionPlan

def build():
    return ProductionPlan.model_validate({
        "title": "كيف يعمل الراوتر؟", "width": 1080, "height": 1920,
        "shots": [
            {"caption":"ما هو الراوتر؟","narration":"الراوتر جهاز يربط أجهزتك بالشبكة ويوجه البيانات بينها.","seconds":5,
             "gesture":"present","camera":"medium","objects":[
                 {"kind":"router","animation":"pop_in","x":0.75,"y":0.39,"start":0.3,"end":5}]},
            {"caption":"التوصيل بالكابل","narration":"كابل الإيثرنت ينقل البيانات بين الأجهزة والراوتر عبر اتصال سلكي.","seconds":5,
             "gesture":"point","camera":"wide","objects":[
                 {"kind":"router","animation":"static","x":0.76,"y":0.35},
                 {"kind":"ethernet_cable","animation":"slide_in","x":0.74,"y":0.59,"start":0.7,"end":5}]},
            {"caption":"اتصال الواي فاي","narration":"وبالواي فاي يرسل الراوتر إشارات لاسلكية تتصل بها الهواتف والحواسيب.","seconds":6,
             "gesture":"explain","camera":"wide","objects":[
                 {"kind":"router","animation":"static","x":0.76,"y":0.34},
                 {"kind":"wifi_signal","animation":"pulse","x":0.76,"y":0.18,"start":0.4,"end":6},
                 {"kind":"phone","animation":"pop_in","x":0.75,"y":0.64,"start":1.3,"end":6}]}
        ]
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build().model_dump_json(indent=2), encoding="utf-8")
    print(path)
