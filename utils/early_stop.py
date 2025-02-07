from collections import Counter

def most_frequent_elements(lst):
    if not lst:
        return []  # 如果列表为空，返回空列表或其他适当的值
    lst = [x for x in lst if x is not None]
    if not lst:
        return []  # 如果列表为空，返回空列表或其他适当的值
    counter = Counter(lst)
    max_count = max(counter.values())  # 找到最高的出现次数
    # 过滤出所有出现次数等于最高次数的元素
    most_common_elements = [item for item, count in counter.items() if count == max_count]
    return most_common_elements

# def most_frequent_elements(activated_answers_list, lst):
#     if not lst:
#         return []  # 如果列表为空，返回空列表或其他适当的值
#     max_num_id = -1
#     import pdb; pdb.set_trace()
#     for idx, v in enumerate(lst):
#         if isinstance(lst[max_num_id], list) and len(v) > len(lst[max_num_id]) and v[0] > 60:
#             max_num_id = idx

#     # flattened_scores = [score for idx, sublist in enumerate(lst) for score in sublist]
#     # if all(x == 50 for x in flattened_scores):
#     #     max_num_id = -1

#     return activated_answers_list[max_num_id]

def highest_score_elements(scores):
    flattened_scores_with_index = [(score, idx) for idx, sublist in enumerate(scores) for score in sublist]
    if not flattened_scores_with_index:
        return []
    try:
        max_score, max_index = max(flattened_scores_with_index, key=lambda x: x[0])
    except:
        import pdb; pdb.set_trace()
    # 由于我们可能有多个子列表具有相同的最高分，我们找出所有这样的子列表
    all_max_indices = [idx for score, idx in flattened_scores_with_index if score == max_score]
    return all_max_indices
