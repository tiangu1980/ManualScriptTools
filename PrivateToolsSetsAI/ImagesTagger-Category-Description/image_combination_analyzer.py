
import json
import os
from collections import defaultdict
from itertools import combinations, product

class ImageCombinationGenerator:
    def __init__(self, json_file):
        print("Initializing ImageCombinationGenerator...")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file {json_file} does not exist.")
        self.json_file = json_file
        self.data = self.load_data()
        self.combination_rules = {
            'animals': (0, 1),
            'terrain': (1, 3),
            'weather': (1, 2),
            'mood': (1, 2),
            'plants': (1, 2),
            'scenery': (1, 3)
        }
        self.output_dir = "combinations"
        self.combination_id = 1
        self.stats = []
        
    def load_data(self):
        print("Loading data from JSON file...")
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_unique_values(self):
        values = {
            'countries': set(),
            'regions': set(),
            'category_labels': defaultdict(set)
        }
        
        for item in self.data:
            values['countries'].add(item['location']['country'])
            values['regions'].add(item['location']['region'])
            for tag in item['tags']:
                values['category_labels'][tag['category']].add(tag['label'])
        
        return {
            'countries': sorted(values['countries']),
            'regions': sorted(values['regions']),
            'category_labels': {k: sorted(v) for k, v in values['category_labels'].items() 
                              if k in self.combination_rules}
        }
    
    def generate_combinations(self):
        print("Generating combinations...")
        unique_values = self.get_unique_values()
        
        # 生成地理位置组合
        loc_combinations = []
        for country in unique_values['countries'] + ['']:
            for region in unique_values['regions'] + ['']:
                if country or region:
                    loc_combinations.append((country, region))
        loc_combinations.append(('', ''))  # 无地理信息
        
        # 生成分类标签组合
        category_combs = {}
        for category, labels in unique_values['category_labels'].items():
            min_len, max_len = self.combination_rules[category]
            combs = []
            for n in range(max(1, min_len), max_len + 1):
                combs.extend(combinations(labels, n))
            category_combs[category] = combs
        
        # 生成所有可能的组合
        for loc_comb in loc_combinations:
            country, region = loc_comb
            
            # 选择1到所有分类的组合
            for category_count in range(1, len(self.combination_rules) + 1):
                for selected_categories in combinations(self.combination_rules.keys(), category_count):
                    # 生成选中分类的标签组合
                    label_combs = [category_combs[cat] for cat in selected_categories]
                    for label_comb in product(*label_combs):
                        self.process_combination(country, region, selected_categories, label_comb)
    
    def process_combination(self, country, region, categories, label_comb):
        print(f"Processing combination: {country}, {region}, {categories}, {label_comb}")
        # 创建组合描述
        desc_parts = []
        if country:
            desc_parts.append(f'In country: "{country}"')
        if region:
            desc_parts.append(f'In region: "{region}"')
        
        category_desc = {}
        for category, labels in zip(categories, label_comb):
            label_str = ', '.join(labels)
            if category == 'animals':
                category_desc['animals'] = f'animals "{label_str}"'
            else:
                category_desc[category] = f'{category} is "{label_str}"'
        
        # 按固定顺序添加分类描述
        for cat in ['terrain', 'animals', 'plants', 'weather', 'mood', 'scenery']:
            if cat in category_desc:
                desc_parts.append(category_desc[cat])
        
        # 查找匹配的图片
        matched_images = []
        for item in self.data:
            # 检查地理位置
            loc_match = True
            if country and item['location']['country'] != country:
                loc_match = False
            if region and item['location']['region'] != region:
                loc_match = False
            if not loc_match:
                continue
            
            # 检查分类标签
            category_match = True
            for category, labels in zip(categories, label_comb):
                found = False
                for tag in item['tags']:
                    if tag['category'] == category and tag['label'] in labels:
                        found = True
                        break
                if not found:
                    category_match = False
                    break
            
            if category_match:
                matched_images.append(item['image'])
        
        if matched_images:
            # 保存匹配的图片列表
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            with open(f"{self.output_dir}/{self.combination_id}.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(matched_images))
            
            # 添加到统计列表
            combination_desc = ' ; '.join(desc_parts)
            self.stats.append(f"{self.combination_id}  {combination_desc} = {len(matched_images)}")
            self.combination_id += 1
    
    def save_statistics(self):
        print("Saving statistics...")
        with open('combination_stats.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(self.stats)))
    
    def analyze(self):
        print("Starting analysis...")
        self.generate_combinations()
        self.save_statistics()
        print(f"分析完成！共生成 {len(self.stats)} 种有效组合")
        print(f"组合统计已保存到 combination_stats.txt")
        print(f"每种组合的图片列表已保存到 {self.output_dir}/ 目录")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python image_combination_analyzer.py <json文件>")
        sys.exit(1)
    
    analyzer = ImageCombinationGenerator(sys.argv[1])
    analyzer.analyze()
