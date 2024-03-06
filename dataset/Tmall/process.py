def read_meta_data():

    # 序列推荐中需要用到padding，将padding定位为id为0的物品
    item_cnt, taxonomy_cnt = 1, 0

    with open('./tianchi_2014001_rec_tmall_product.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            splited = line.strip().split('\x01')
            # 字符串类型
            # 3-hop: parent category, child category, brand
            old_item, category, brand = splited[0], splited[-3], splited[-2]
            p_category, c_category = category.split('-')

            if old_item not in item_old2new:
                item_old2new[old_item] = item_cnt
                item_cnt += 1
            new_item = item_old2new[old_item]

            old_taxonomies = [p_category, c_category, brand]
            new_taxonomies = []
            for old_taxonomy in old_taxonomies:
                if old_taxonomy not in taxonomy_old2new:
                    taxonomy_old2new[old_taxonomy] = taxonomy_cnt
                    taxonomy_cnt += 1

                new_taxonomy = taxonomy_old2new[old_taxonomy]
                new_taxonomies.append(new_taxonomy)

            if new_item not in item_taxonomy_dict:
                item_taxonomy_dict[new_item] = new_taxonomies


def read_interact_data():
    pass


if __name__ == '__main__':

    # 现在的问题是，是将三跳的类别全部映射到一起，还是分开映射
    # 先尝试全部映射到一起，不行再分开映射

    # key: old item id      value: new item id
    item_old2new = dict()
    # key: old taxonomy id      value: new taxonomy id
    taxonomy_old2new = dict()
    # key: item     value: [p_category, c_category, brand]
    item_taxonomy_dict = dict()

    read_meta_data()
    read_interact_data()
