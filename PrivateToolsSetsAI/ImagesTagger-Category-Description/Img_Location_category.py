import os
import json
import re
from PIL import Image, ExifTags
from llama_cpp import Llama

# 初始化本地模型（路径请改为你本地的模型文件）
llm = Llama(
    model_path="./modelsmistral-7b-instruct.gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # 模型路径
    n_ctx=2048,
    n_threads=8,    # 根据你机器的核心数调整
    n_gpu_layers=0  # CPU运行；若你支持GPU可以调高
)

# 清理描述中的特殊字符
def clean_description(description: str) -> str:
    """
    替换 description 中的特殊字符为单个空格，保留英文、数字、空格和常见字母符号。
    """
    if not description:
        return ""

    # 用正则表达式替换所有非字母数字和空格的字符为一个空格
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", description)

    # 再将多个连续空格压缩成一个
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()

# 从图像中提取 ImageDescription
def get_image_description(img_path):
    try:
        image = Image.open(img_path)
        exif_data = image._getexif()
        if not exif_data:
            return ""
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "ImageDescription":
                str_value = value if isinstance(value, str) else str(value, 'utf-8', errors='ignore')
                value = clean_description(str_value)
                return value.strip()
        return ""
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return ""

# 简单关键词映射（可扩展）
fallback_keywords = {
    "USA": ["Grand Canyon", "Yellowstone", "Yosemite", "Statue of Liberty", "Times Square", "Las Vegas", "Golden Gate Bridge", "Hawaii", "Mount Rushmore", "Niagara Falls", "Central Park", "Hollywood", "Miami", "New York", "Los Angeles", "San Francisco", "Washington D.C."],
    "Canada": ["Niagara Falls", "Banff", "Jasper", "Whistler", "Toronto", "Vancouver", "Montreal", "CN Tower", "Quebec City", "Ottawa"],
    "Mexico": ["Cancun", "Chichen Itza", "Tulum", "Mexico City", "Teotihuacan", "Cabo San Lucas", "Guadalajara", "Puebla"],
    "Brazil": ["Christ the Redeemer", "Sugarloaf Mountain", "Iguazu Falls", "Amazon Rainforest", "Rio de Janeiro", "Sao Paulo", "Brasilia"],
    "UK": ["London", "Big Ben", "Stonehenge", "Edinburgh", "Loch Ness", "Lake District", "Windsor Castle", "Cambridge", "Oxford"],
    "France": ["Eiffel Tower", "Louvre", "Mont Saint-Michel", "Versailles", "Nice", "French Riviera", "Chamonix", "Provence", "Normandy"],
    "Italy": ["Rome", "Colosseum", "Venice", "Florence", "Amalfi Coast", "Cinque Terre", "Pisa", "Vatican", "Milan", "Naples"],
    "Spain": ["Barcelona", "Sagrada Familia", "Madrid", "Alhambra", "Seville", "Ibiza", "Granada", "Valencia", "Toledo"],
    "Germany": ["Berlin", "Neuschwanstein", "Munich", "Black Forest", "Cologne Cathedral", "Hamburg", "Frankfurt", "Dresden"],
    "Switzerland": ["Zermatt", "Matterhorn", "Jungfrau", "Lucerne", "Interlaken", "Lake Geneva", "Zurich", "Bern"],
    "Greece": ["Santorini", "Athens", "Acropolis", "Mykonos", "Crete", "Delphi", "Rhodes", "Thessaloniki"],
    "Turkey": ["Istanbul", "Cappadocia", "Pamukkale", "Ephesus", "Antalya", "Mount Ararat", "Ankara"],
    "Russia": ["Moscow", "Red Square", "Saint Petersburg", "Lake Baikal", "Kremlin", "Trans-Siberian Railway", "Sochi", "Novosibirsk"],
    "India": ["Taj Mahal", "Jaipur", "Kerala", "Himalayas", "Goa", "Varanasi", "Delhi", "Mumbai", "Agra", "Udaipur"],
    "China": ["Great Wall", "Beijing", "Shanghai", "Terracotta Army", "Guilin", "Zhangjiajie", "Tibet", "Yellow Mountains", "Xi'an", "Hangzhou"],
    "Japan": ["Mount Fuji", "Tokyo", "Kyoto", "Nara", "Osaka", "Himeji", "Hokkaido", "Okinawa", "Hiroshima"],
    "Thailand": ["Phuket", "Bangkok", "Chiang Mai", "Ayutthaya", "Krabi", "Koh Samui", "Phi Phi Islands", "Phetchabun", "Phu Tub Berk"],
    "Vietnam": ["Ha Long Bay", "Hoi An", "Hanoi", "Ho Chi Minh", "Da Nang", "Phong Nha", "Sapa", "Hue"],
    "Indonesia": ["Bali", "Jakarta", "Borobudur", "Mount Bromo", "Komodo", "Yogyakarta", "Lombok"],
    "Australia": ["Sydney", "Opera House", "Great Barrier Reef", "Uluru", "Melbourne", "Tasmania", "Blue Mountains", "Brisbane"],
    "New Zealand": ["Milford Sound", "Queenstown", "Rotorua", "Auckland", "Lake Tekapo", "Hobbiton", "Wellington"],
    "South Africa": ["Cape Town", "Table Mountain", "Kruger Park", "Garden Route", "Durban", "Johannesburg", "Pretoria"],
    "Egypt": ["Pyramids of Giza", "Cairo", "Luxor", "Abu Simbel", "Aswan", "Nile River"],
    "UAE": ["Dubai", "Burj Khalifa", "Abu Dhabi", "Sheikh Zayed Mosque", "Palm Jumeirah"],
    "Morocco": ["Marrakech", "Casablanca", "Fes", "Chefchaouen", "Sahara Desert", "Rabat"],
    "Argentina": ["Buenos Aires", "Iguazu Falls", "Patagonia", "Bariloche", "Perito Moreno Glacier", "Mendoza"],
    "Chile": ["Atacama Desert", "Torres del Paine", "Easter Island", "Santiago", "Valparaiso", "Pucon"],
    "Peru": ["Machu Picchu", "Cusco", "Lima", "Lake Titicaca", "Sacred Valley"],
    "Nepal": ["Mount Everest", "Kathmandu", "Pokhara", "Annapurna", "Lumbini"],
    "Iceland": ["Reykjavik", "Blue Lagoon", "Golden Circle", "Vatnajökull", "Jökulsárlón", "Geysir"],
    "Norway": ["Oslo", "Bergen", "Geirangerfjord", "Lofoten", "Trolltunga", "Preikestolen"],
    "Sweden": ["Stockholm", "Gothenburg", "Kiruna", "Abisko", "Lapland"],
    "Finland": ["Helsinki", "Rovaniemi", "Lapland", "Saimaa", "Santa Claus Village"],
    "Portugal": ["Lisbon", "Porto", "Algarve", "Madeira", "Sintra", "Azores"],
    "Czech Republic": ["Prague", "Český Krumlov", "Karlovy Vary", "Kutná Hora"],
    "Austria": ["Vienna", "Salzburg", "Hallstatt", "Innsbruck", "Alps"],
    "Poland": ["Warsaw", "Krakow", "Wroclaw", "Gdansk"],
    "Netherlands": ["Amsterdam", "Rotterdam", "Keukenhof", "Utrecht"],
    "Belgium": ["Brussels", "Bruges", "Antwerp", "Ghent"],
    "Ireland": ["Dublin", "Cliffs of Moher", "Galway", "Belfast"],
    "Hungary": ["Budapest", "Lake Balaton", "Eger"],
    "South Korea": ["Seoul", "Busan", "Jeju Island", "Gyeongju"],
    "Philippines": ["Manila", "Boracay", "Palawan", "Cebu"],
    "Malaysia": ["Kuala Lumpur", "Penang", "Langkawi", "Borneo"],
    "Singapore": ["Marina Bay Sands", "Sentosa", "Gardens by the Bay"],
    "Colombia": ["Bogota", "Cartagena", "Medellin"],
    "Costa Rica": ["San Jose", "Arenal Volcano", "Monteverde"],
    "Jamaica": ["Kingston", "Negril", "Montego Bay"],
    "Cuba": ["Havana", "Varadero", "Trinidad"],
    # 更多国家和著名景点可以继续扩充
}

# 构建关键词倒排表（只执行一次）
keyword_to_location = {}
for country, places in fallback_keywords.items():
    for place in places:
        keyword_to_location[place.lower()] = {"country": country, "region": place}


def get_location_from_description(description: str) -> dict:
    if not description:
        return {"country": "No Info", "region": "No Info"}

#    prompt = f"""
#你是一名地理专家。请根据以下图像描述，判断这张图片最可能是在哪个国家和地区拍摄的，并严格用 JSON 方式返回结果。
#
#图像描述如下：
#\"\"\"{description}\"\"\"
#
#请**只输出**下面格式的合法 JSON（不要包含任何注释、图片链接、说明或 markdown）：
#{{
#  "country": "国家英文名",
#  "region": "地区英文名"
#}}
#
#如果无法判断，请把字段值设为 ""。
#"""

    prompt = f"""
YOU ARE A PROFESSIONAL GEOGRAPHY EXPERT. BASED ON THE FOLLOWING IMAGE DESCRIPTION, DETERMINE THE MOST LIKELY COUNTRY AND REGION WHERE THIS IMAGE WAS TAKEN, AND RETURN THE RESULT STRICTLY IN JSON FORMAT.
AS A GEOSPATIAL AI, YOU MUST STICTLY FOLLOW THESE RULES:

1. INPUT PROCESSING
   - ANALYZE ONLY: \"\"\"{description}\"\"\"
   - IGNORE ALL EXTERNAL CONTEXT

2. OUTPUT REQUIREMENTS
   - RETURN ONLY THIS JSON:
   {{
     "country": "", 
     "region": ""
   }}

3. DECISION RULES
   - IF MULTIPLE LOCATIONS POSSIBLE:
     * SELECT MOST PROBABLE ONE
     * USE EMPTY STRING IF NO CLEAR WINNER
   - MUST MATCH ISO 3166 NAMES EXACTLY

4. VALIDATION
   - MUST PASS json.loads()
   - FAILURE = {{"country": "", "region": ""}} IF:
     * NO GEOGRAPHIC CLUES
     * MULTIPLE POSSIBLE LOCATIONS
     * REGION UNCLEAR

5. FORMATTING
   - ASCII ONLY
   - NO SPECIAL CHARACTERS
   - NO LINE BREAKS INSIDE JSON

6. PROHIBITIONS
   - NO EXAMPLES (EXCEPT THESE)
   - NO MARKDOWN
   - NO COMMENTS
   - NO PARTIAL RESPONSES

INPUT: \"\"\"{description}\"\"\"
OUTPUT: {{"country": "", "region": ""}}

EXAMPLE:
\"\"\"mountain with pagoda\"\"\" -> {{"country": "", "region": ""}}
\"\"\"Eiffel Tower in Paris\"\"\" -> {{"country": "France", "region": "Ile-de-France"}}
"""



    try:
        response = llm(prompt, stop=["}"], temperature=0.7, top_p=0.9, max_tokens=256)
        text = response["choices"][0]["text"]

        cleaned = text.replace("```json", "").replace("```", "").strip()
        if cleaned.count("{") > cleaned.count("}"):
            cleaned += "}"

        match = re.search(r"\{\s*\"country\"\s*:\s*\"(.*?)\"\s*,\s*\"region\"\s*:\s*\"(.*?)\"\s*\}", cleaned, re.DOTALL)

        print(f"Model response:\n{cleaned}")
        print(f"Matched JSON: {match.group() if match else 'None'}")

        if match:
            country = match.group(1).strip()
            region = match.group(2).strip()

            if not country and not region:
                # fallback
                desc_lower = description.lower()
                for country_key, places in fallback_keywords.items():
                    if country_key.lower() in desc_lower:
                        # 尝试找地区
                        matched_region = None
                        for place in places:
                            if place.lower() in desc_lower:
                                matched_region = place
                                break
                        if matched_region:
                            print(f"🔁 fallback matched country+region: {country_key}, {matched_region}")
                            return {"country": country_key, "region": matched_region}
                        else:
                            print(f"🔁 fallback matched country only: {country_key}, region No Info")
                            return {"country": country_key, "region": "No Info"}

            return {
                "country": country or "No Info",
                "region": region or "No Info"
            }

        else:
            # 正则没匹配到，走fallback逻辑
            desc_lower = description.lower()
            for country_key, places in fallback_keywords.items():
                if country_key.lower() in desc_lower:
                    matched_region = None
                    for place in places:
                        if place.lower() in desc_lower:
                            matched_region = place
                            break
                    if matched_region:
                        print(f"🔁 fallback matched country+region: {country_key}, {matched_region}")
                        return {"country": country_key, "region": matched_region}
                    else:
                        print(f"🔁 fallback matched country only: {country_key}, region No Info")
                        return {"country": country_key, "region": "No Info"}

            return {"country": "No Info", "region": "No Info"}

    except Exception as e:
        print(f"❌ 解析模型输出失败: {e}\n原始文本:\n{locals().get('text', '[no text]')}")
        desc_lower = description.lower()
        for country_key, places in fallback_keywords.items():
            if country_key.lower() in desc_lower:
                matched_region = None
                for place in places:
                    if place.lower() in desc_lower:
                        matched_region = place
                        break
                if matched_region:
                    print(f"🔁 fallback matched country+region: {country_key}, {matched_region}")
                    return {"country": country_key, "region": matched_region}
                else:
                    print(f"🔁 fallback matched country only: {country_key}, region No Info")
                    return {"country": country_key, "region": "No Info"}

        return {"country": "No Info", "region": "No Info"}


    
# 主函数：处理所有图片
def process_images_in_directory(directory_path, output_json_path="total_locations_v2.json"):
    result = []
    supported_exts = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(supported_exts):
            continue

        img_path = os.path.join(directory_path, filename)
        print(f"!!! Processing image: {img_path}")
        description = get_image_description(img_path)
        print(f"    Extracted description: {description}")
        location = {"country": "No Info", "region": "No Info"}
        if description:
            location = get_location_from_description(description)
        print(f"    Extracted location for {filename}: {location}")
        result.append({
            "image": filename,
            "location": {
                "country": location.get("country", ""),
                "region": location.get("region", "")
            }
        })
        print(f"✅ Finished processing {filename}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"📄 结果已保存到 {output_json_path}")

# 示例执行
if __name__ == "__main__":
    image_folder = "D:\\GrowthAI\\4 ReGroupInDimensions\\Total"  # 替换为你的图片目录
    process_images_in_directory(image_folder)
