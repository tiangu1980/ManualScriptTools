
import pandas as pd
import os

def generate_html(excel_path, output_path):
    df = pd.read_excel(excel_path)
    categories = sorted(df['Category'].value_counts().to_dict().items())
    #sorted_categories = sorted(categories, key=lambda x: x[1], reverse=True)  # 按File排序
    
    print("Category    File")
    for cat, count in categories:
        print(f"{cat}    {count}")
    
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>图片分类浏览器</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        .image-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;}
        .image-card {border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden; display: none;}
        .image-card img {width: 100%; height: 180px; object-fit: contain;}
        .card-body {padding: 15px;}
        .select-container {margin: 20px 0;}
        .select-label {margin-right: 10px; font-weight: bold;}
    </style>
</head>
<body>
    <div class="container">
        <h2 class="my-4">图片分类浏览器</h2>
        <div class="select-container">
            <span class="select-label">筛选分类:</span>
            <select id="categoryFilter" class="form-select" multiple>
'''
    html_content += f'''
                <option value="all" selected>全部显示 (共{len(df)}张)</option>
'''

    for cat, count in categories:
        html_content += f'<option value="{cat}">{cat} ({count}张)</option>'

    html_content += '''
            </select>
            <small class="text-muted">按住Ctrl键可多选</small>
        </div>
        <div class="image-grid">'''

    for _, row in df.iterrows():
        html_content += f'''
            <div class="image-card" data-category="{row['Category']}">
                <img src="{row['File']}" class="card-img-top" alt="{row['File']}">
                <div class="card-body">
                    <h5 class="card-title">分类: {row['Category']}</h5>
                    <p class="card-text">文件: {row['File']}</p>
                </div>
            </div>'''

    html_content += '''
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('categoryFilter').addEventListener('change', function() {
            const selected = Array.from(this.selectedOptions).map(opt => opt.value);
            const cards = document.querySelectorAll('.image-card');
            
            if(selected.includes('all') || selected.length === 0) {
                cards.forEach(card => card.style.display = 'block');
            } else {
                cards.forEach(card => {
                    card.style.display = selected.includes(card.dataset.category) 
                        ? 'block' 
                        : 'none';
                });
            }
        });

        window.onload = function() {
            document.querySelectorAll('.image-card').forEach(card => {
                card.style.display = 'block';
            });
        };
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML文件已生成: {output_path}")

if __name__ == "__main__":
    excel_path = input("请输入Excel文件路径: ")
    base_path = excel_path.replace('.xlsx', '.html')
    output_path = base_path
    #output_path = input("请输入HTML输出路径: ")
    print(output_path)
    generate_html(excel_path, output_path)
