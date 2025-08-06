
import pandas as pd
import joblib

def load_model():
    model = joblib.load('uninstall_reason_classifier.pkl')
    catname_encoder = joblib.load('category_mapping.pkl')
    reverse_mapping = {v: k for k, v in catname_encoder.items()}
    return model, reverse_mapping

def batch_predict(input_file, output_file):
    # 读取输入文件
    df = pd.read_csv(input_file).fillna('Unknown') 
    
    # 加载模型
    model, reverse_mapping = load_model()
    
    # 进行预测
    pred_ids = model.predict(df['Uninstallreason'])
    df['Catogorie'] = pred_ids
    df['CatogorieName'] = [reverse_mapping[id] for id in pred_ids]
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == '__main__':
    input_csv = input("请输入输入文件路径: ")
    output_csv = input("请输入输出文件路径: ")
    batch_predict(input_csv, output_csv)
