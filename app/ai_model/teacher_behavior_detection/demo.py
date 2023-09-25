import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'src/auto_text_classifier'))
from src.teacher_behavior_detection import detect


if __name__ == "__main__":
    input_text = [
        
        {
            "text": "请你总结一下这节课的内容",
            "begin_time": 1326752,
            "end_time": 1332165
        },
        {
            "text": "总结归纳一下文章主旨",
            "begin_time": 1326752,
            "end_time": 1332165
        }
    ]

    # 测试用
    for i in range(1):
        result = detect(input_text, keywords_scene='qingqing')
        print(result['data']['text_result'])
        # print(json.dumps(result, indent=4, ensure_ascii=False))
