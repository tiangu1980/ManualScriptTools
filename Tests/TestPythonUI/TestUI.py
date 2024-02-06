import tkinter as tk
import subprocess

def run_script():
    script_path = "your_script.py"
    arg1 = entry_arg1.get()
    arg2 = entry_arg2.get()
    subprocess.run(["python", script_path, arg1, arg2])

# 创建主窗口
root = tk.Tk()
root.title("Python Script Runner")

# 创建输入框和标签
label_arg1 = tk.Label(root, text="Argument 1:")
label_arg1.pack()
entry_arg1 = tk.Entry(root)
entry_arg1.pack()

label_arg2 = tk.Label(root, text="Argument 2:")
label_arg2.pack()
entry_arg2 = tk.Entry(root)
entry_arg2.pack()

# 创建运行按钮
button_run = tk.Button(root, text="Run Script", command=run_script)
button_run.pack()

# 运行主事件循环
root.mainloop()
