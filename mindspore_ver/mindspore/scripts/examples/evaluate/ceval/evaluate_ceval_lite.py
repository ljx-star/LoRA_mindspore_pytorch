import os
from typing import List
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore.common import set_seed
from mindformers import LlamaTokenizer
from mindformers.inference import InferConfig, InferTask
from mindformers.generation.utils import softmax

'''
数据地址
https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
'''

task2desc = {
    "high_school_physics": "高中物理",
    "fire_engineer": "注册消防工程师",
    "computer_network": "计算机网络",
    "advanced_mathematics": "高等数学",
    "logic": "逻辑学",
    "middle_school_physics": "初中物理",
    "clinical_medicine": "临床医学",
    "probability_and_statistics": "概率统计",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "operating_system": "操作系统",
    "middle_school_mathematics": "初中数学",
    "chinese_language_and_literature": "中国语言文学",
    "electrical_engineer": "注册电气工程师",
    "business_administration": "工商管理",
    "high_school_geography": "高中地理",
    "modern_chinese_history": "近代史纲要",
    "legal_professional": "法律职业资格",
    "middle_school_geography": "初中地理",
    "middle_school_chemistry": "初中化学",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "physician": "医师资格",
    "high_school_chinese": "高中语文",
    "tax_accountant": "税务师",
    "high_school_history": "高中历史",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
    "high_school_mathematics": "高中数学",
    "professional_tour_guide": "导游资格",
    "veterinary_medicine": "兽医学",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "basic_medicine": "基础医学",
    "education_science": "教育学",
    "urban_and_rural_planner": "注册城乡规划师",
    "middle_school_biology": "初中生物",
    "plant_protection": "植物保护",
    "middle_school_history": "初中历史",
    "high_school_politics": "高中政治",
    "metrology_engineer": "注册计量师",
    "art_studies": "艺术学",
    "college_economics": "大学经济学",
    "college_chemistry": "大学化学",
    "law": "法学",
    "sports_science": "体育学",
    "civil_servant": "公务员",
    "college_programming": "大学编程",
    "middle_school_politics": "初中政治",
    "teacher_qualification": "教师资格",
    "computer_architecture": "计算机组成",
    "college_physics": "大学物理",
    "discrete_mathematics": "离散数学",
    "marxism": "马克思主义基本原理",
    "accountant": "注册会计师",
}

def load_models_tokenizer(args):
    tokenizer = LlamaTokenizer(args.token_path)

    lite_config = InferConfig(
        prefill_model_path=args.full_model_path,
        increment_model_path=args.inc_model_path,
        model_type="mindir",
        model_name="llama",
        ge_config_path=args.config_path,
        device_id=args.device_id,
        infer_seq_length=4096,
    )

    pipeline_task = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)

    return pipeline_task, tokenizer


def format_example(line, subject, include_answer=True):
    example = f"以下是中国关于{task2desc[subject]}考试的单项选择题，请选出其中的正确答案。\n\n"
    example = example + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["answer"] + "\n\n"
    else:
        example += "\n答案："
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    prompt = ""
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            subject,
            include_answer=True,
        )
    return prompt


def get_logits(tokenizer, pipeline_task, inputs: List[str]):
    input_ids = tokenizer(inputs, padding="max_length", max_length=4096, truncation=True, truncate_direction="LEFT")["input_ids"]

    valid_length = []
    valid_length.append(np.max(np.argwhere(np.array(input_ids[0]) != tokenizer.pad_token_id)) + 1)
    valid_length = np.array(valid_length, np.int32)
    current_index = [valid_length[0] - 1]
    current_index = np.array(current_index, np.int32)
    input_ids = np.array(input_ids, np.int32)
    lite_inputs = pipeline_task.get_predict_inputs(pipeline_task.full_model, input_ids, current_index)
    outputs = pipeline_task.full_model.predict(lite_inputs)
    outputs = outputs[0].get_data_to_numpy()
    outputs = np.array([outputs[0][current_index[0], :]])
    logits = softmax(outputs, axis=-1)

    return logits, inputs


def eval_subject(
        model,
        tokenizer,
        subject_name,
        test_df,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        **kwargs,
):
    file_path = os.path.join(save_result_dir, f"{subject_name}_result.csv") if save_result_dir else None
    if file_path and os.path.exists(file_path):
        # Read the file, extract the 'correctness' column, and calculate correct_ratio
        existing_df = pd.read_csv(file_path, encoding="utf-8")
        if "correctness" in existing_df:
            correct_ratio = 100 * existing_df["correctness"].sum() / len(existing_df["correctness"])
            return correct_ratio

    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else ""
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, subject_name, include_answer=False)
        full_prompt = few_shot_prompt + question

        output, input_info = get_logits(tokenizer, model, [full_prompt])
        assert output.shape[0] == 1
        logits = output.flatten()

        softval = softmax(np.asarray(
            [
                logits[tokenizer("A")["input_ids"][-1]],
                logits[tokenizer("B")["input_ids"][-1]],
                logits[tokenizer("C")["input_ids"][-1]],
                logits[tokenizer("D")["input_ids"][-1]],
            ]),
            axis=0,
        )
        if softval.dtype in {np.float16}:
            softval = softval.to(dtype=np.float32)
        probs = softval

        for i, choice in enumerate(choices):
            all_probs[f"prob_{choice}"].append(probs[i])
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)

    if score:
        correct_ratio = 100 * sum(score) / len(score)
        if args.debug:
            print(subject_name, correct_ratio)
    else:
        correct_ratio = 0
    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return correct_ratio


def cal_ceval(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.0
    for tt in res.keys():
        name = tt.split("-")[-1]
        acc_sum += float(res[tt])
        cnt += 1
        class_ = TASK_NAME_MAPPING[name][2]
        if class_ not in acc_sum_dict:
            acc_sum_dict[class_] = 0.0
            acc_norm_sum_dict[class_] = 0.0
            cnt_dict[class_] = 0.0
        if name in hard_list:
            hard_cnt += 1
            hard_acc_sum += float(res[tt])
        acc_sum_dict[class_] += float(res[tt])
        cnt_dict[class_] += 1
    print("\n\n\n")
    for k in ["STEM", "Social Science", "Humanities", "Other"]:
        if k in cnt_dict:
            print("%s acc: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k]))
    if hard_cnt > 0:
        print("Hard acc:%.2f " % (hard_acc_sum / hard_cnt))
    print("AVERAGE acc:%.2f " % (acc_sum / cnt))


TASK_NAME_MAPPING = {
    "computer_network": ["Computer Network", "\u8ba1\u7b97\u673a\u7f51\u7edc", "STEM"],
    "operating_system": ["Operating System", "\u64cd\u4f5c\u7cfb\u7edf", "STEM"],
    "computer_architecture": [
        "Computer Architecture",
        "\u8ba1\u7b97\u673a\u7ec4\u6210",
        "STEM",
    ],
    "college_programming": ["College Programming", "\u5927\u5b66\u7f16\u7a0b", "STEM"],
    "college_physics": ["College Physics", "\u5927\u5b66\u7269\u7406", "STEM"],
    "college_chemistry": ["College Chemistry", "\u5927\u5b66\u5316\u5b66", "STEM"],
    "advanced_mathematics": [
        "Advanced Mathematics",
        "\u9ad8\u7b49\u6570\u5b66",
        "STEM",
    ],
    "probability_and_statistics": [
        "Probability and Statistics",
        "\u6982\u7387\u7edf\u8ba1",
        "STEM",
    ],
    "discrete_mathematics": [
        "Discrete Mathematics",
        "\u79bb\u6563\u6570\u5b66",
        "STEM",
    ],
    "electrical_engineer": [
        "Electrical Engineer",
        "\u6ce8\u518c\u7535\u6c14\u5de5\u7a0b\u5e08",
        "STEM",
    ],
    "metrology_engineer": [
        "Metrology Engineer",
        "\u6ce8\u518c\u8ba1\u91cf\u5e08",
        "STEM",
    ],
    "high_school_mathematics": [
        "High School Mathematics",
        "\u9ad8\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "high_school_physics": ["High School Physics", "\u9ad8\u4e2d\u7269\u7406", "STEM"],
    "high_school_chemistry": [
        "High School Chemistry",
        "\u9ad8\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "high_school_biology": ["High School Biology", "\u9ad8\u4e2d\u751f\u7269", "STEM"],
    "middle_school_mathematics": [
        "Middle School Mathematics",
        "\u521d\u4e2d\u6570\u5b66",
        "STEM",
    ],
    "middle_school_biology": [
        "Middle School Biology",
        "\u521d\u4e2d\u751f\u7269",
        "STEM",
    ],
    "middle_school_physics": [
        "Middle School Physics",
        "\u521d\u4e2d\u7269\u7406",
        "STEM",
    ],
    "middle_school_chemistry": [
        "Middle School Chemistry",
        "\u521d\u4e2d\u5316\u5b66",
        "STEM",
    ],
    "veterinary_medicine": ["Veterinary Medicine", "\u517d\u533b\u5b66", "STEM"],
    "college_economics": [
        "College Economics",
        "\u5927\u5b66\u7ecf\u6d4e\u5b66",
        "Social Science",
    ],
    "business_administration": [
        "Business Administration",
        "\u5de5\u5546\u7ba1\u7406",
        "Social Science",
    ],
    "marxism": [
        "Marxism",
        "\u9a6c\u514b\u601d\u4e3b\u4e49\u57fa\u672c\u539f\u7406",
        "Social Science",
    ],
    "mao_zedong_thought": [
        "Mao Zedong Thought",
        "\u6bdb\u6cfd\u4e1c\u601d\u60f3\u548c\u4e2d\u56fd\u7279\u8272\u793e\u4f1a\u4e3b\u4e49\u7406\u8bba\u4f53\u7cfb\u6982\u8bba",
        "Social Science",
    ],
    "education_science": ["Education Science", "\u6559\u80b2\u5b66", "Social Science"],
    "teacher_qualification": [
        "Teacher Qualification",
        "\u6559\u5e08\u8d44\u683c",
        "Social Science",
    ],
    "high_school_politics": [
        "High School Politics",
        "\u9ad8\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "high_school_geography": [
        "High School Geography",
        "\u9ad8\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "middle_school_politics": [
        "Middle School Politics",
        "\u521d\u4e2d\u653f\u6cbb",
        "Social Science",
    ],
    "middle_school_geography": [
        "Middle School Geography",
        "\u521d\u4e2d\u5730\u7406",
        "Social Science",
    ],
    "modern_chinese_history": [
        "Modern Chinese History",
        "\u8fd1\u4ee3\u53f2\u7eb2\u8981",
        "Humanities",
    ],
    "ideological_and_moral_cultivation": [
        "Ideological and Moral Cultivation",
        "\u601d\u60f3\u9053\u5fb7\u4fee\u517b\u4e0e\u6cd5\u5f8b\u57fa\u7840",
        "Humanities",
    ],
    "logic": ["Logic", "\u903b\u8f91\u5b66", "Humanities"],
    "law": ["Law", "\u6cd5\u5b66", "Humanities"],
    "chinese_language_and_literature": [
        "Chinese Language and Literature",
        "\u4e2d\u56fd\u8bed\u8a00\u6587\u5b66",
        "Humanities",
    ],
    "art_studies": ["Art Studies", "\u827a\u672f\u5b66", "Humanities"],
    "professional_tour_guide": [
        "Professional Tour Guide",
        "\u5bfc\u6e38\u8d44\u683c",
        "Humanities",
    ],
    "legal_professional": [
        "Legal Professional",
        "\u6cd5\u5f8b\u804c\u4e1a\u8d44\u683c",
        "Humanities",
    ],
    "high_school_chinese": [
        "High School Chinese",
        "\u9ad8\u4e2d\u8bed\u6587",
        "Humanities",
    ],
    "high_school_history": [
        "High School History",
        "\u9ad8\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "middle_school_history": [
        "Middle School History",
        "\u521d\u4e2d\u5386\u53f2",
        "Humanities",
    ],
    "civil_servant": ["Civil Servant", "\u516c\u52a1\u5458", "Other"],
    "sports_science": ["Sports Science", "\u4f53\u80b2\u5b66", "Other"],
    "plant_protection": ["Plant Protection", "\u690d\u7269\u4fdd\u62a4", "Other"],
    "basic_medicine": ["Basic Medicine", "\u57fa\u7840\u533b\u5b66", "Other"],
    "clinical_medicine": ["Clinical Medicine", "\u4e34\u5e8a\u533b\u5b66", "Other"],
    "urban_and_rural_planner": [
        "Urban and Rural Planner",
        "\u6ce8\u518c\u57ce\u4e61\u89c4\u5212\u5e08",
        "Other",
    ],
    "accountant": ["Accountant", "\u6ce8\u518c\u4f1a\u8ba1\u5e08", "Other"],
    "fire_engineer": [
        "Fire Engineer",
        "\u6ce8\u518c\u6d88\u9632\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "environmental_impact_assessment_engineer": [
        "Environmental Impact Assessment Engineer",
        "\u73af\u5883\u5f71\u54cd\u8bc4\u4ef7\u5de5\u7a0b\u5e08",
        "Other",
    ],
    "tax_accountant": ["Tax Accountant", "\u7a0e\u52a1\u5e08", "Other"],
    "physician": ["Physician", "\u533b\u5e08\u8d44\u683c", "Other"],
}
hard_list = [
    "advanced_mathematics",
    "discrete_mathematics",
    "probability_and_statistics",
    "college_physics",
    "college_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
]
choices = ["A", "B", "C", "D"]


def main(args):
    print(args)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)
    pipeline_task, tokenizer = load_models_tokenizer(args)

    dev_result = {}
    for subject_name in tqdm(TASK_NAME_MAPPING.keys()):
        val_file_path = os.path.join(
            args.eval_data_path, "val", f"{subject_name}_val.csv"
        )
        dev_file_path = os.path.join(
            args.eval_data_path, "dev", f"{subject_name}_dev.csv"
        )
        # test_file_path = os.path.join(args.eval_data_path, 'test', f'{subject_name}_test.csv')
        val_df = pd.read_csv(val_file_path)
        dev_df = pd.read_csv(dev_file_path)
        # test_df = pd.read_csv(test_file_path)

        score = eval_subject(
            pipeline_task,
            tokenizer,
            subject_name,
            val_df,
            dev_df=dev_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/ceval_eval_result_lite",
        )
        dev_result[subject_name] = score
    cal_ceval(dev_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint path", default="")
    parser.add_argument("-t", "--token_path", type=str, help="Tokenizer.model path", default="")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device_id", type=int, default=0, help="Device id")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, required=True, help="Path to eval data")
    group.add_argument("--max-seq-len", type=int, default=2048, help="Size of the output generated text.")
    group.add_argument("--debug", action="store_true", default=False, help="Print infos.")
    group.add_argument("--config", type=str, required=False, help="Path to config")
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    group.add_argument("--config_path", type=str, required=False, help="Path to GE config")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
