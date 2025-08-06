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
    .radio-list {{ margin-bottom: 20px; }}
    .theme-block {{ display: none; margin-bottom: 40px; }}
    .theme-block img {{ width: 200px; height: auto; margin: 5px; }}
    .theme-title {{ font-size: 20px; font-weight: bold; }}
    .theme-description {{ margin: 5px 0 10px 0; }}
    .theme-keywords {{ font-style: italic; color: #555; }}
  </style>
</head>
<body>

<h1>Image Themes Viewer</h1>
<p><strong>Total unique images: {unique_count}</strong></p>

<div class="radio-list">
  <label><input type="radio" name="theme" value="none" checked> Show None</label><br>
  {radio_buttons}
</div>

<div id="themes-container">
  {theme_blocks}
</div>

<script>
  const radios = document.querySelectorAll('input[type=radio][name=theme]');
  const blocks = document.querySelectorAll('.theme-block');

  radios.forEach(rb => {{
    rb.addEventListener('change', () => {{
      blocks.forEach(block => block.style.display = 'none');
      if (rb.value !== 'none') {{
        document.getElementById(rb.value).style.display = 'block';
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
    radio_html = []
    theme_blocks_html = []
    all_image_set = set()

    for idx, theme in enumerate(themes):
        theme_id = f"theme_{idx}_{sanitize_id(theme['theme'])}"
        title = theme["title"]
        description = theme["description"]
        keywords = theme["theme"]
        images = theme["images"]
        img_count = len(images)
        all_image_set.update(images)

        # Radio button
        radio_html.append(f'''
        <label>
          <input type="radio" name="theme" value="{theme_id}"> {title} - {img_count} imgs
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
          <div class="theme-keywords">{keywords}</div>
          <div class="theme-images">{image_tags}</div>
        </div>
        ''')

    html_content = HTML_TEMPLATE.format(
        unique_count=len(all_image_set),
        radio_buttons="\n".join(radio_html),
        theme_blocks="\n".join(theme_blocks_html)
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML written to: {output_path} (Total unique images: {len(all_image_set)})")

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
