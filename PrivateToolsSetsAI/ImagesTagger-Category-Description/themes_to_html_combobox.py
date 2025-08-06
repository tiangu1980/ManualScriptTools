import json
import argparse
import os

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Image Themes</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .checkbox-list {{ margin-bottom: 20px; }}
    .theme-block {{ display: none; margin-bottom: 40px; }}
    .theme-block img {{ width: 200px; height: auto; margin: 5px; }}
    .theme-title {{ font-size: 20px; font-weight: bold; }}
    .theme-description {{ margin: 5px 0 10px 0; }}
  </style>
</head>
<body>

<h1>Image Themes Viewer</h1>

<div class="checkbox-list">
  {checkboxes}
</div>

<div id="themes-container">
  {theme_blocks}
</div>

<script>
  const checkboxes = document.querySelectorAll('input[type=checkbox]');
  checkboxes.forEach(cb => {{
    cb.addEventListener('change', () => {{
      const block = document.getElementById(cb.value);
      if (cb.checked) {{
        block.style.display = 'block';
      }} else {{
        block.style.display = 'none';
      }}
    }});
  }});
</script>

</body>
</html>
"""

def sanitize_id(name):
    return ''.join(c if c.isalnum() else '_' for c in name)

def generate_html(themes, output_path):
    checkbox_html = []
    theme_blocks_html = []

    for idx, theme in enumerate(themes):
        theme_id = f"theme_{idx}_{sanitize_id(theme['theme'])}"
        title = theme["theme"]
        description = theme["description"]
        images = theme["images"]
        img_count = len(images)

        # Checkbox
        checkbox_html.append(f'''
        <label>
          <input type="checkbox" value="{theme_id}"> {title} - {img_count} imgs
        </label><br>
        ''')

        # Image block
        image_tags = '\n'.join(
            [f'<img src="{os.path.basename(img)}" alt="{title}">' for img in images]
        )

        theme_blocks_html.append(f'''
        <div class="theme-block" id="{theme_id}">
          <div class="theme-title">{title} ({img_count} images)</div>
          <div class="theme-description">{description}</div>
          <div class="theme-images">{image_tags}</div>
        </div>
        ''')

    html_content = HTML_TEMPLATE.format(
        checkboxes="\n".join(checkbox_html),
        theme_blocks="\n".join(theme_blocks_html)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert themes JSON to interactive HTML")
    parser.add_argument("input_json", help="Path to clustered themes JSON")
    parser.add_argument("output_html", help="Path to output HTML file")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        themes = json.load(f)

    generate_html(themes, args.output_html)

if __name__ == "__main__":
    main()
