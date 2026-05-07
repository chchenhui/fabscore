# Echo: Multi-agent AI system for patient-centered pharmacovigilance data exploration interface

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import pandas as pd
from collections import defaultdict
import re

class PharmcovigilanceInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Pharmacovigilance Data Explorer - Echo System")
        self.root.geometry("1600x900")
        self.root.configure(bg='#FFF8E8')
        
        self.colors = {
            'cream': '#FFF8E8',
            'sage': '#DEE8CE', 
            'terracotta': '#BB6653',
            'orange': '#F08B51',
            'dark_gray': '#2D3748',
            'medium_gray': '#4A5568',
            'light_gray': '#718096'
        }
        
        self.setup_styles()
        
        self.raw_data = {}
        self.processed_data = []
        self.filtered_data = []
        self.sort_column = None
        self.sort_reverse = False
        
        self.create_interface()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', 
                       font=('Inter', 28, 'bold'), 
                       background=self.colors['cream'], 
                       foreground=self.colors['dark_gray'])
        
        style.configure('Subtitle.TLabel', 
                       font=('Inter', 16), 
                       background=self.colors['cream'], 
                       foreground=self.colors['medium_gray'])
        
        style.configure('Search.TEntry', 
                       fieldbackground='white', 
                       borderwidth=2, 
                       relief='flat', 
                       font=('Inter', 14),
                       focuscolor=self.colors['orange'])
        
        style.configure('Data.Treeview', 
                       background='white',
                       foreground=self.colors['dark_gray'],
                       fieldbackground='white',
                       borderwidth=0,
                       font=('Inter', 13),
                       rowheight=40)
        
        style.configure('Data.Treeview.Heading',
                       background=self.colors['terracotta'],
                       foreground='white',
                       font=('Inter', 14, 'bold'),
                       borderwidth=0,
                       relief='flat',
                       padding=(10, 15))
        
        style.configure('Vertical.TScrollbar',
                       background=self.colors['sage'],
                       troughcolor=self.colors['cream'],
                       borderwidth=0,
                       arrowcolor=self.colors['terracotta'])
        
        style.configure('Horizontal.TScrollbar',
                       background=self.colors['sage'],
                       troughcolor=self.colors['cream'],
                       borderwidth=0,
                       arrowcolor=self.colors['terracotta'])
        
    def create_interface(self):
        header_frame = tk.Frame(self.root, bg=self.colors['cream'], pady=40)
        header_frame.pack(fill='x')
        
        title_label = ttk.Label(header_frame, 
                               text="Echo: A multi-agent AI system for patient-centered pharmacovigilance",
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame,
                                  text="Discovering unreported oncology drug side effects from Reddit discussions",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(10, 0))
        
        control_frame = tk.Frame(self.root, bg=self.colors['cream'], pady=20)
        control_frame.pack(fill='x', padx=40)
        
        load_button = tk.Button(control_frame, 
                               text="📁 Load JSON Data", 
                               command=self.load_data, 
                               bg=self.colors['orange'], 
                               fg='white', 
                               relief='flat',
                               font=('Inter', 14, 'bold'), 
                               padx=25, 
                               pady=15,
                               borderwidth=0,
                               activebackground=self.colors['terracotta'], 
                               activeforeground='white')
        load_button.pack(side='left', padx=(0, 30))
        
        search_frame = tk.Frame(control_frame, bg=self.colors['cream'])
        search_frame.pack(side='left', fill='x', expand=True)
        
        search_label = tk.Label(search_frame, 
                               text="🔍 Search:", 
                               background=self.colors['cream'], 
                               foreground=self.colors['dark_gray'],
                               font=('Inter', 14, 'bold'))
        search_label.pack(side='left', padx=(0, 15))
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search)
        
        search_container = tk.Frame(search_frame, bg='white', relief='flat', bd=2)
        search_container.pack(side='left', padx=(0, 20))
        
        search_entry = tk.Entry(search_container, 
                               textvariable=self.search_var,
                               font=('Inter', 14),
                               width=35,
                               relief='flat',
                               bd=0,
                               bg='white',
                               fg=self.colors['dark_gray'])
        search_entry.pack(padx=15, pady=12)
        
        clear_button = tk.Button(search_frame, 
                                text="✕ Clear", 
                                command=self.clear_search,
                                bg=self.colors['sage'], 
                                fg=self.colors['dark_gray'], 
                                relief='flat',
                                font=('Inter', 12, 'bold'), 
                                padx=20, 
                                pady=10,
                                borderwidth=0,
                                activebackground=self.colors['terracotta'], 
                                activeforeground='white')
        clear_button.pack(side='left')
        
        self.results_label = tk.Label(control_frame, 
                                     text="No data loaded", 
                                     background=self.colors['cream'], 
                                     foreground=self.colors['medium_gray'],
                                     font=('Inter', 14))
        self.results_label.pack(side='right')
        
        self.create_table()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load JSON data to begin analysis")
        
        status_bar = tk.Frame(self.root, bg=self.colors['sage'], height=50)
        status_bar.pack(side='bottom', fill='x', pady=(20, 0))
        status_bar.pack_propagate(False)
        
        status_label = tk.Label(status_bar, 
                               textvariable=self.status_var,
                               bg=self.colors['sage'], 
                               fg=self.colors['dark_gray'],
                               font=('Inter', 13))
        status_label.pack(expand=True, pady=15)
        
    def create_table(self):
        table_container = tk.Frame(self.root, bg=self.colors['cream'])
        table_container.pack(fill='both', expand=True, padx=40, pady=(0, 30))
        
        table_frame = tk.Frame(table_container, bg='white', relief='flat', borderwidth=0)
        table_frame.pack(fill='both', expand=True)
        
        shadow_frame = tk.Frame(table_container, bg='#E2E8F0', height=4)
        shadow_frame.pack(fill='x', pady=(0, 1))
        
        columns = ('Drug', 'Symptom', 'Temporal', 'Confidence', 
                  'Community', 'Novelty Score', 'Analyze', 'Proposer')
        
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                                style='Data.Treeview', height=16)
        
        column_widths = {
            'Drug': 200, 
            'Symptom': 250, 
            'Avg Temporal': 140,
            'Avg Confidence': 150, 
            'Avg Community': 150,
            'Novelty Score': 140,
            'Analyzer Report': 160, 
            'Proposer Report': 160
        }
        
        self.numerical_columns = {'Avg Temporal', 'Avg Confidence', 'Avg Community', 'Novelty Score'}
        
        for col in columns:
            self.tree.heading(col, text=col, anchor='center', 
                            command=lambda c=col: self.sort_by_column(c))
            self.tree.column(col, width=column_widths.get(col, 140), 
                           anchor='center', minwidth=120)
        
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        v_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 2), pady=2)
        h_scrollbar.grid(row=1, column=0, sticky='ew', padx=2, pady=(0, 2))
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.tree.bind('<Double-1>', self.on_cell_double_click)
        
    def sort_by_column(self, column):
        if not self.filtered_data:
            return
            
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_reverse = False
            
        self.sort_column = column
        
        column_map = {
            'Drug': 'drug',
            'Symptom': 'symptom',
            'Avg Temporal': 'avg_temporal',
            'Avg Confidence': 'avg_confidence',
            'Avg Community': 'avg_community',
            'Novelty Score': 'novelty_score'
        }
        
        if column in column_map:
            key = column_map[column]
            
            if column in self.numerical_columns:
                self.filtered_data.sort(key=lambda x: x[key], reverse=self.sort_reverse)
            else:
                self.filtered_data.sort(key=lambda x: x[key].lower(), reverse=self.sort_reverse)
                
            self.populate_table()
            
            for col in ['Drug', 'Symptom', 'Avg Temporal', 'Avg Confidence', 'Avg Community', 'Novelty Score']:
                if col == column:
                    direction = " ▼" if self.sort_reverse else " ▲"
                    self.tree.heading(col, text=col + direction)
                else:
                    self.tree.heading(col, text=col)
        
    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select JSON Data File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            self.process_data()
            self.populate_table()
            self.status_var.set(f"✅ Loaded {len(self.processed_data)} drug-symptom associations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def process_data(self):
        self.processed_data = []
        
        for drug, symptoms in self.raw_data.items():
            for symptom, entries in symptoms.items():
                if symptom == 'null':
                    continue
                    
                if entries:
                    avg_temporal = sum(entry.get('temporal_weight', 0) for entry in entries) / len(entries)
                    avg_confidence = sum(entry.get('confidence', 0) for entry in entries) / len(entries)
                    avg_community = sum(entry.get('community_metric', 0) for entry in entries) / len(entries)
                    
                    novelty_score = (avg_temporal + avg_confidence + avg_community) / 3
                    
                    all_confounders = []
                    all_quotes = []
                    
                    for entry in entries:
                        all_confounders.extend(entry.get('confounders', []))
                        if entry.get('quote'):
                            all_quotes.append(entry['quote'])
                    
                    self.processed_data.append({
                        'drug': drug,
                        'symptom': symptom,
                        'avg_temporal': round(avg_temporal, 3),
                        'avg_confidence': round(avg_confidence, 3),
                        'avg_community': round(avg_community, 3),
                        'novelty_score': round(novelty_score, 3),
                        'confounders': list(set(all_confounders)),
                        'quotes': all_quotes,
                        'raw_entries': entries
                    })
        
        self.processed_data.sort(key=lambda x: x['novelty_score'], reverse=True)
        self.filtered_data = self.processed_data.copy()
        
    def populate_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, row in enumerate(self.filtered_data):
            values = (
                row['drug'],
                row['symptom'],
                f"{row['avg_temporal']:.3f}",
                f"{row['avg_confidence']:.3f}",
                f"{row['avg_community']:.3f}",
                f"{row['novelty_score']:.3f}",
                "📊 Analysis",
                "📋 Report"
            )
            
            tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
            self.tree.insert('', 'end', values=values, tags=tags)
        
        self.tree.tag_configure('evenrow', background='#FEFEFE')
        self.tree.tag_configure('oddrow', background='#F8FAFC')
        
        self.update_results_count()
        
    def on_search(self, *args):
        search_term = self.search_var.get().lower().strip()
        
        if not search_term:
            self.filtered_data = self.processed_data.copy()
        else:
            self.filtered_data = [
                row for row in self.processed_data
                if (search_term in row['drug'].lower() or 
                    search_term in row['symptom'].lower())
            ]
        
        self.populate_table()
        
    def clear_search(self):
        self.search_var.set("")
        
    def update_results_count(self):
        total = len(self.processed_data)
        filtered = len(self.filtered_data)
        
        if total == 0:
            self.results_label.config(text="No data loaded")
        elif total == filtered:
            self.results_label.config(text=f"📈 Showing {total:,} associations")
        else:
            self.results_label.config(text=f"📈 Showing {filtered:,} of {total:,} associations")
            
    def create_modern_button(self, parent, text, command, bg_color, fg_color='white', font_size=14):
        button = tk.Button(parent, text=text, command=command,
                          bg=bg_color, fg=fg_color, relief='flat',
                          font=('Inter', font_size, 'bold'), 
                          padx=25, pady=15,
                          activebackground=self.colors['terracotta'],
                          activeforeground='white', borderwidth=0)
        return button
            
    def on_cell_double_click(self, event):
        item = self.tree.selection()[0]
        column = self.tree.identify_column(event.x)
        
        row_index = self.tree.index(item)
        if row_index >= len(self.filtered_data):
            return
            
        row_data = self.filtered_data[row_index]
        
        if column == '#7':
            self.show_analyzer_report(row_data)
        elif column == '#8':
            self.show_proposer_report(row_data)
            
    def show_analyzer_report(self, row_data):
        popup = tk.Toplevel(self.root)
        popup.title(f"Analyzer Report - {row_data['drug']} / {row_data['symptom']}")
        popup.geometry("900x700")
        popup.configure(bg='white')
        popup.transient(self.root)
        popup.grab_set()
        
        header_frame = tk.Frame(popup, bg=self.colors['terracotta'], pady=35)
        header_frame.pack(fill='x')
        
        title_label = tk.Label(header_frame, 
                              text=f"📊 {row_data['drug']} → {row_data['symptom']}",
                              font=('Inter', 20, 'bold'),
                              bg=self.colors['terracotta'], fg='white')
        title_label.pack()
        
        content_frame = tk.Frame(popup, bg='white', padx=40, pady=30)
        content_frame.pack(fill='both', expand=True)
        
        metrics_card = tk.Frame(content_frame, bg=self.colors['cream'], 
                               relief='flat', bd=0)
        metrics_card.pack(fill='x', pady=(0, 25), padx=10, ipady=20)
        
        metrics_title = tk.Label(metrics_card, text="📈 Analysis Metrics", 
                                font=('Inter', 16, 'bold'), 
                                bg=self.colors['cream'],
                                fg=self.colors['dark_gray'])
        metrics_title.pack(pady=(15, 10))
        
        metrics_text = f"""Temporal Weight: {row_data['avg_temporal']:.3f}
Confidence Score: {row_data['avg_confidence']:.3f}
Community Metric: {row_data['avg_community']:.3f}
Novelty Score: {row_data['novelty_score']:.3f}
Number of Reports: {len(row_data['raw_entries'])}"""
        
        metrics_label = tk.Label(metrics_card, text=metrics_text, 
                                font=('Inter', 14),
                                bg=self.colors['cream'], 
                                justify='left', 
                                fg=self.colors['medium_gray'])
        metrics_label.pack(anchor='w', padx=30, pady=(0, 15))
        
        confounders_card = tk.Frame(content_frame, bg=self.colors['sage'], 
                                   relief='flat', bd=0)
        confounders_card.pack(fill='x', pady=(0, 25), padx=10, ipady=20)
        
        confounders_title = tk.Label(confounders_card, text="⚠️ Identified Confounders",
                                    font=('Inter', 16, 'bold'), 
                                    bg=self.colors['sage'],
                                    fg=self.colors['dark_gray'])
        confounders_title.pack(pady=(15, 10))
        
        if row_data['confounders']:
            confounders_text = "• " + "\n• ".join(row_data['confounders'])
        else:
            confounders_text = "No confounders identified"
            
        confounders_label = tk.Label(confounders_card, text=confounders_text, 
                                    font=('Inter', 13),
                                    bg=self.colors['sage'], 
                                    justify='left', wraplength=750, 
                                    fg=self.colors['medium_gray'])
        confounders_label.pack(anchor='w', padx=30, pady=(0, 15))
        
        quotes_title = tk.Label(content_frame, text="💬 Supporting Evidence",
                               font=('Inter', 16, 'bold'), 
                               bg='white',
                               fg=self.colors['dark_gray'])
        quotes_title.pack(anchor='w', pady=(0, 15))
        
        quotes_frame = tk.Frame(content_frame, bg='white')
        quotes_frame.pack(fill='both', expand=True)
        
        quotes_text = tk.Text(quotes_frame, font=('Inter', 13),
                             bg='#F8FAFC', wrap='word', height=8, 
                             padx=20, pady=15,
                             borderwidth=2, relief='flat', 
                             selectbackground=self.colors['orange'],
                             selectforeground='white',
                             fg=self.colors['dark_gray'])
        quotes_scrollbar = ttk.Scrollbar(quotes_frame, command=quotes_text.yview)
        quotes_text.config(yscrollcommand=quotes_scrollbar.set)
        
        for i, quote in enumerate(row_data['quotes'], 1):
            quotes_text.insert('end', f"Quote {i}:\n{quote}\n\n")
        
        quotes_text.config(state='disabled')
        
        quotes_text.pack(side='left', fill='both', expand=True)
        quotes_scrollbar.pack(side='right', fill='y')
        
        button_frame = tk.Frame(popup, bg='white', pady=20)
        button_frame.pack(fill='x')
        
        close_button = self.create_modern_button(button_frame, "✕ Close", 
                                                popup.destroy, self.colors['terracotta'])
        close_button.pack()
        
    def show_proposer_report(self, row_data):
        popup = tk.Toplevel(self.root)
        popup.title(f"Proposer Report - {row_data['drug']} / {row_data['symptom']}")
        popup.geometry("800x600")
        popup.configure(bg='white')
        popup.transient(self.root)
        popup.grab_set()
        
        header_frame = tk.Frame(popup, bg=self.colors['orange'], pady=35)
        header_frame.pack(fill='x')
        
        title_label = tk.Label(header_frame, 
                              text=f"🤖 AI-Generated Insights",
                              font=('Inter', 20, 'bold'),
                              bg=self.colors['orange'], fg='white')
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame,
                                 text=f"{row_data['drug']} → {row_data['symptom']}",
                                 font=('Inter', 14),
                                 bg=self.colors['orange'], fg='white')
        subtitle_label.pack(pady=(8, 0))
        
        content_frame = tk.Frame(popup, bg='white', padx=40, pady=30)
        content_frame.pack(fill='both', expand=True)
        
        dummy_report = f"""
DRUG-INDUCED SYMPTOM ANALYSIS

PRIMARY HYPOTHESIS: NEUROINFLAMMATORY CASCADE
The compound may accumulate in brain regions critical for affected neural pathways. The drug's known propensity to cause peripheral effects through inflammatory cascades may extend to the central nervous system. Neuroinflammation in relevant brain networks could disrupt neural circuits responsible for the observed symptom pattern. This mechanism is supported by the drug's documented ability to cross physiological barriers and cause systemic effects.

SECONDARY HYPOTHESIS: CELLULAR DYSFUNCTION PATHWAY
The therapeutic agent may contribute to symptom manifestation through oxidative stress and mitochondrial dysfunction in target neurons. The drug's interference with cellular energy metabolism could preferentially affect high-energy-demand regions involved in the affected physiological processes, leading to the observed clinical presentation. This aligns with existing literature showing similar therapeutic classes can cause persistent effects affecting related biological systems.
"""
        
        text_widget = tk.Text(content_frame, font=('Inter', 13),
                             bg=self.colors['cream'], wrap='word', 
                             padx=25, pady=20,
                             borderwidth=2, relief='flat', 
                             selectbackground=self.colors['orange'],
                             selectforeground='white',
                             fg=self.colors['dark_gray'])
        text_widget.insert('1.0', dummy_report.strip())
        text_widget.config(state='disabled')
        text_widget.pack(fill='both', expand=True)
        
        button_frame = tk.Frame(popup, bg='white', pady=20)
        button_frame.pack(fill='x')
        
        close_button = self.create_modern_button(button_frame, "✕ Close", 
                                                popup.destroy, self.colors['terracotta'])
        close_button.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = PharmcovigilanceInterface(root)
    root.mainloop()