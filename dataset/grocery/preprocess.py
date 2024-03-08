import pandas as pd
import json
import pickle


# 对于不在用户交互数据里的那些物品，不需要读取n_hop数据
def read_meta_data(n_hop=2):
    # item从1开始，0充当padding
    # 为了统一，taxonomy也从1开始(或者taxonomy也需要padding。即taxonomy不足目标数时，利用padding代替)
    taxonomy_cnt = 1
    taxonomy2id['0'] = 0

    with open('meta_Grocery_and_Gourmet_Food.json', 'r') as f:
        for line in f.readlines():
            item_meta_dict = json.loads(line)

            # 对于categories而言，需要除去第一大类，因为他们的第一大类都一样
            # 除去第一大类后，种类数大于等于1
            old_item, categories = item_meta_dict['asin'], item_meta_dict['category'][1:]

            if old_item not in item_old2new:
                continue
            new_item = item_old2new[old_item]

            # # len(categories) + len(['0']) + len(brand) == n_hop & len(brand) == 1
            # if len(categories) > n_hop - 2:
            #     old_taxonomies = categories[:n_hop - 1] + [brand]
            # else:
            #     old_taxonomies = categories + ['0'] * (n_hop - 1 - len(categories)) + [brand]

            # 先把品牌去掉，品牌加入后会出现不能构建成树的问题
            old_taxonomies = []
            idx = 0
            while len(old_taxonomies) < n_hop and idx < len(categories):
                old_taxonomies.append(categories[idx])
                idx += 1

            # 不足则补0
            while len(old_taxonomies) < n_hop:
                old_taxonomies.append('0')

            new_taxonomies = []
            for old_taxonomy in old_taxonomies:
                if old_taxonomy not in taxonomy2id:
                    taxonomy2id[old_taxonomy] = taxonomy_cnt
                    taxonomy_cnt += 1

                new_taxonomy = taxonomy2id[old_taxonomy]
                new_taxonomies.append(new_taxonomy)

            if new_item not in item_taxonomy_dict:
                item_taxonomy_dict[new_item] = new_taxonomies


def read_interact_data():
    # 为了方便，user_cnt和item_cnt统一开始计数
    user_cnt, item_cnt = 1, 1

    users, items, timestamps = [], [], []
    # 该数据集为5-core，即少于五条交互数据的用户和物品都被去除了
    # 并且只要是用户评价过的，都将转换为隐式反馈
    with open('Grocery_and_Gourmet_Food_5.json', 'r') as f:
        for line in f.readlines():
            interact_dict = json.loads(line)

            old_user, old_item, old_timestamp = interact_dict['reviewerID'], interact_dict['asin'], interact_dict[
                'reviewTime']

            # 如果该物品没有元数据
            if old_item in difference:
                continue

            if old_user not in user_old2new:
                user_old2new[old_user] = user_cnt
                user_cnt += 1
            new_user = user_old2new[old_user]

            if old_item not in item_old2new:
                item_old2new[old_item] = item_cnt
                item_cnt += 1
            new_item = item_old2new[old_item]

            # 将时间按照年月日拼接，可以比大小
            splited = old_timestamp.strip().split(',')
            month_day, year = splited[0], splited[1]
            month, day = month_day.split(' ')

            # day前面可能需要补0
            if len(day) == 1:
                day = '0' + day

            new_timestamp = int(year.strip() + month + day)

            users.append(new_user)
            items.append(new_item)
            timestamps.append(new_timestamp)

    df = pd.DataFrame({
        'user': users,
        'item': items,
        'timestamp': timestamps
    })

    # 排序，先根据user排序，再根据timestamp进行排序
    df.sort_values(by=['user', 'timestamp'], inplace=True)

    # 将df写进新文件中
    target_file = 'grocery.txt'
    # 去除timestamp这一列
    new_df = df.iloc[:, :2]
    new_df.to_csv(target_file, sep=' ', index=False, header=False)

    print(f'user nums: {len(user_old2new)}, item nums: {len(item_old2new)}')
    print(f'average length: {len(df) / len(user_old2new)}')


# 有些物品可能存在交互数据中，但是没有元数据
def get_difference():
    all_items = set()
    with open('meta_Grocery_and_Gourmet_Food.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = json.loads(line)['asin']
            all_items.add(item)

    interact_items = set()
    with open('Grocery_and_Gourmet_Food_5.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = json.loads(line)['asin']
            interact_items.add(item)

    return list(interact_items.difference(all_items))


if __name__ == '__main__':

    # 获取差集：列表元素为存在于交互数据中，但没有元数据的物品
    difference = get_difference()

    # 目前存在疑惑是，对于不同层次的taxonomy，是将其映射到一起还是每个层级单独映射
    item_old2new, user_old2new = dict(), dict()
    read_interact_data()

    taxonomy2id = dict()
    item_taxonomy_dict = dict()

    # 对于
    read_meta_data(n_hop=2)
    # 这里会报错，问题在于交互数据中的部分物品没有metadata，需要将这一部分物品删除
    # 确保交互数据中的每个物品都有元数据
    assert len(item_taxonomy_dict) == len(item_old2new)

    # 保存树状结构
    # 其余保存后面会用到，用于解释
    with open('item_taxonomy.dict', 'wb') as f:
        pickle.dump(item_taxonomy_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('taxonomy2id.json', 'w') as f:
        json.dump(taxonomy2id, f, indent=4)
