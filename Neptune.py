import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import os
import time


# Define the APIClientApp class
class APIClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neptune")

        # Set light teal background color
        self.root.configure(bg='#e0f7fa')

        self.history_file = "api_history.json"
        self.load_history()

        # URL Entry
        self.url_label = ttk.Label(root, text="URL:", background='#e0f7fa')
        self.url_label.grid(column=0, row=0, padx=10, pady=5, sticky='w')
        self.url_entry = ttk.Entry(root, width=50)
        self.url_entry.grid(column=1, row=0, padx=10, pady=5, sticky='ew')

        # Method Dropdown
        self.method_label = ttk.Label(root, text="Method:", background='#e0f7fa')
        self.method_label.grid(column=0, row=1, padx=10, pady=5, sticky='w')
        self.method_var = tk.StringVar()
        self.method_dropdown = ttk.Combobox(root, textvariable=self.method_var)
        self.method_dropdown['values'] = ('GET', 'POST', 'PUT', 'DELETE')
        self.method_dropdown.grid(column=1, row=1, padx=10, pady=5, sticky='ew')
        self.method_dropdown.current(0)

        # Content Type Dropdown
        self.content_type_label = ttk.Label(root, text="Content Type:", background='#e0f7fa')
        self.content_type_label.grid(column=0, row=2, padx=10, pady=5, sticky='w')
        self.content_type_var = tk.StringVar()
        self.content_type_dropdown = ttk.Combobox(root, textvariable=self.content_type_var)
        self.content_type_dropdown['values'] = ('application/json', 'application/x-www-form-urlencoded')
        self.content_type_dropdown.grid(column=1, row=2, padx=10, pady=5, sticky='ew')
        self.content_type_dropdown.current(0)
        self.content_type_dropdown.bind("<<ComboboxSelected>>", self.update_body_field)

        # Headers Entry
        self.headers_label = ttk.Label(root, text="Headers:", background='#e0f7fa')
        self.headers_label.grid(column=0, row=3, padx=10, pady=5, sticky='w')
        self.headers_frame = ttk.Frame(root)
        self.headers_frame.grid(column=1, row=3, padx=10, pady=5, sticky='ew')
        self.headers = []
        self.add_header_row(default=True)

        # Body Entry
        self.body_label = ttk.Label(root, text="Body:", background='#e0f7fa')
        self.body_label.grid(column=0, row=4, padx=10, pady=5, sticky='w')
        self.body_entry = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
        self.body_entry.grid(column=1, row=4, padx=10, pady=5, sticky='ew')

        # HTTP Response Code and Time
        self.response_code_label = ttk.Label(root, text="HTTP Response Code:", background='#e0f7fa')
        self.response_code_label.grid(column=0, row=5, padx=10, pady=5, sticky='w')
        self.response_code_value = ttk.Label(root, text="", background='#e0f7fa')
        self.response_code_value.grid(column=1, row=5, padx=10, pady=5, sticky='w')

        self.response_time_label = ttk.Label(root, text="Total Time:", background='#e0f7fa')
        self.response_time_label.grid(column=0, row=6, padx=10, pady=5, sticky='w')
        self.response_time_value = ttk.Label(root, text="", background='#e0f7fa')
        self.response_time_value.grid(column=1, row=6, padx=10, pady=5, sticky='w')

        # Response Tabs
        self.response_tabs = ttk.Notebook(root)
        self.response_tabs.grid(column=1, row=7, padx=10, pady=5, sticky='nsew')

        self.response_text_tab = ttk.Frame(self.response_tabs)
        self.response_headers_tab = ttk.Frame(self.response_tabs)

        self.response_tabs.add(self.response_text_tab, text='Response Text')
        self.response_tabs.add(self.response_headers_tab, text='Response Headers')

        self.response_text_display = scrolledtext.ScrolledText(self.response_text_tab, width=50, height=10, wrap=tk.WORD)
        self.response_text_display.pack(fill='both', expand=True)

        self.response_headers_display = scrolledtext.ScrolledText(self.response_headers_tab, width=50, height=10, wrap=tk.WORD)
        self.response_headers_display.pack(fill='both', expand=True)

        # Send Button
        self.send_button = ttk.Button(root, text="Send", command=self.send_request)
        style = ttk.Style()
        style.configure("Send.TButton", background='#4caf50', foreground='black', padding=6, relief="flat")
        style.map("Send.TButton",
              background=[('active', '#45a049')],
              relief=[('pressed', 'sunken')])
        self.send_button.configure(style="Send.TButton")
        self.send_button.grid(column=1, row=8, padx=10, pady=10, sticky='ew')

        # History Listbox
        self.history_label = ttk.Label(root, text="History:", background='#e0f7fa')
        self.history_label.grid(column=2, row=0, padx=10, pady=5, sticky='w')
        self.history_listbox = tk.Listbox(root, width=50, height=20)
        self.history_listbox.grid(column=2, row=1, rowspan=8, padx=10, pady=5, sticky='nsew')
        self.history_listbox.bind('<<ListboxSelect>>', self.load_from_history)

        self.update_history_listbox()

        # Configure grid to scale dynamically
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(7, weight=1)

    def add_header_row(self, default=False):
        row = len(self.headers)
        key_entry = ttk.Entry(self.headers_frame, width=20)
        key_entry.grid(column=0, row=row, padx=5, pady=2, sticky='ew')
        value_entry = ttk.Entry(self.headers_frame, width=30)
        value_entry.grid(column=1, row=row, padx=5, pady=2, sticky='ew')
        remove_button = ttk.Button(self.headers_frame, text="Remove", command=lambda: self.remove_header_row(row))
        remove_button.grid(column=2, row=row, padx=5, pady=2, sticky='ew')
        if row == 0:
            add_button = ttk.Button(self.headers_frame, text="Add Header", command=self.add_header_row)
            add_button.grid(column=3, row=row, padx=5, pady=2, sticky='ew')
        self.headers.append((key_entry, value_entry, remove_button))
        if default:
            key_entry.insert(0, "User-agent")
            value_entry.insert(0, "localhost")

    def remove_header_row(self, row):
        for widget in self.headers[row]:
            widget.grid_forget()
        self.headers.pop(row)
        for i in range(row, len(self.headers)):
            for widget in self.headers[i]:
                widget.grid(row=i)

    def update_body_field(self, event):
        content_type = self.content_type_var.get()
        if content_type == 'application/json':
            self.body_entry = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
            self.body_entry.grid(column=1, row=4, padx=10, pady=5, sticky='ew')
            if hasattr(self, 'body_key_value_frame'):
                self.body_key_value_frame.grid_forget()
        elif content_type == 'application/x-www-form-urlencoded':
            self.body_entry.grid_forget()
            self.body_key_value_frame = ttk.Frame(self.root)
            self.body_key_value_frame.grid(column=1, row=4, padx=10, pady=5, sticky='ew')
            self.body_key_value_entries = []
            self.add_body_key_value_row()
            self.add_body_key_value_fields()

    def add_body_key_value_fields(self):
        self.body_key_value_frame = ttk.Frame(self.root)
        self.body_key_value_frame.grid(column=1, row=4, padx=10, pady=5, sticky='ew')
        self.body_key_value_entries = []
        self.add_body_key_value_row()

    def add_body_key_value_row(self):
        row = len(self.body_key_value_entries)
        key_entry = ttk.Entry(self.body_key_value_frame, width=20)
        key_entry.grid(column=0, row=row, padx=5, pady=2, sticky='ew')
        value_entry = ttk.Entry(self.body_key_value_frame, width=30)
        value_entry.grid(column=1, row=row, padx=5, pady=2, sticky='ew')
        remove_button = ttk.Button(self.body_key_value_frame, text="Remove", command=lambda: self.remove_body_key_value_row(row))
        remove_button.grid(column=2, row=row, padx=5, pady=2, sticky='ew')
        if row == 0:
            add_button = ttk.Button(self.body_key_value_frame, text="Add Field", command=self.add_body_key_value_row)
            add_button.grid(column=3, row=row, padx=5, pady=2, sticky='ew')
        self.body_key_value_entries.append((key_entry, value_entry, remove_button))

    def remove_body_key_value_row(self, row):
        for widget in self.body_key_value_entries[row]:
            widget.grid_forget()
        self.body_key_value_entries.pop(row)
        for i in range(row, len(self.body_key_value_entries)):
            for widget in self.body_key_value_entries[i]:
                widget.grid(row=i)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as file:
                self.history = json.load(file)
        else:
            self.history = []

    def save_history(self):
        with open(self.history_file, 'w') as file:
            json.dump(self.history, file, indent=4)

    def update_history_listbox(self):
        self.history_listbox.delete(0, tk.END)
        for entry in self.history:
            self.history_listbox.insert(tk.END, f"{entry['method']} {entry['url']}")

    def load_from_history(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            entry = self.history[index]
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, entry['url'])
            self.method_var.set(entry['method'])
            for widget in self.headers_frame.winfo_children():
                widget.destroy()
            self.headers = []
            for key, value in entry['headers'].items():
                self.add_header_row()
                self.headers[-1][0].insert(0, key)
                self.headers[-1][1].insert(0, value)
            self.body_entry.delete("1.0", tk.END)
            self.body_entry.insert(tk.END, json.dumps(entry['body'], indent=4))

    def send_request(self):
        url = self.url_entry.get()
        method = self.method_var.get()
        headers = {key.get(): value.get() for key, value, _ in self.headers}
        body = self.body_entry.get("1.0", tk.END).strip()

        try:
            headers = json.dumps(headers) if headers else {}
            body = json.loads(body) if body else {}

            start_time = time.time()

            if method == 'GET':
                response = requests.get(url, headers=json.loads(headers))
            elif method == 'POST':
                content_type = json.loads(headers).get('Content-Type', '')
                if content_type == 'application/x-www-form-urlencoded':
                    response = requests.post(url, headers=json.loads(headers), data=body)
                else:
                    response = requests.post(url, headers=json.loads(headers), json=body)
            elif method == 'PUT':
                response = requests.put(url, headers=json.loads(headers), json=body)
            elif method == 'DELETE':
                response = requests.delete(url, headers=json.loads(headers))

            end_time = time.time()
            total_time = end_time - start_time

            self.response_code_value.config(text=str(response.status_code))
            self.response_time_value.config(text=f"{total_time * 1000:.2f} ms" if total_time < 1 else f"{total_time:.3f} seconds")

            self.response_text_display.delete("1.0", tk.END)
            self.response_text_display.insert(tk.END, json.dumps(response.json(), indent=4))

            self.response_headers_display.delete("1.0", tk.END)
            self.response_headers_display.insert(tk.END, json.dumps(dict(response.headers), indent=4))

            # Save to history
            self.history.append({
                'url': url,
                'method': method,
                'headers': json.loads(headers),
                'body': body
            })
            self.save_history()
            self.update_history_listbox()
        except Exception as e:
            self.response_code_value.config(text="Error")
            self.response_time_value.config(text="")
            self.response_text_display.delete("1.0", tk.END)
            self.response_text_display.insert(tk.END, str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = APIClientApp(root)
    root.mainloop()