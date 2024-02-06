import sys

def main():
    # 检查参数数量
    if len(sys.argv) != 3:
        print("Usage: python your_script.py arg1 arg2")
        return

    # 获取参数
    arg1 = float(sys.argv[1])
    arg2 = float(sys.argv[2])

    # 计算和并输出
    result = arg1 + arg2
    print("Result:", result)

if __name__ == "__main__":
    main()
