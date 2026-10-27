# 📚 Advanced Retrieval Augmented Generation

مشروع **RAG متقدم** للإجابة على الأسئلة من كتاب PDF، باستخدام **Gemini** ونظام بحث هجين (**Qdrant + BM25**)، مع واجهة محادثة بسيطة مبنية بـ **Streamlit**.

يقبل المشروع الأسئلة بالعربية، ويبحث داخل الكتاب بالإنجليزية، ثم يجيب بلغة السؤال.

---

## ✨ المميزات

- **Query Rewriting:** إعادة صياغة سؤال المستخدم إلى استعلام بحث واحد واضح بالإنجليزية.
- **Query Expansion:** توليد 5 استعلامات بديلة بمصطلحات مختلفة لرفع فرصة العثور على الإجابة.
- **Hybrid Search:** دمج البحث بالمعنى (Vector Search عبر Qdrant) مع البحث بالكلمات (BM25) بوزن 50% لكل منهما.
- **Metadata Filtering:** تحديد البحث في نطاق صفحات معيّن من الكتاب باستخدام `Filter` و`Range` في Qdrant.
- **Streaming:** تظهر إجابة Gemini تدريجيًا أثناء كتابتها.
- **شفافية الخطوات:** تعرض الواجهة الاستعلام المُعاد صياغته والاستعلامات الموسّعة.

---

## 🔄 آلية العمل

```
سؤال المستخدم (عربي)
        ↓
Query Rewriting   →  استعلام واحد بالإنجليزية
        ↓
Query Expansion   →  5 استعلامات إضافية
        ↓
لكل استعلام: Hybrid Search
   ├─ BM25                      (مطابقة الكلمات)
   └─ Qdrant + MiniLM           (مطابقة المعنى)  ← فلتر الصفحات (اختياري)
        ↓
دمج النتائج وإزالة التكرار
        ↓
Gemini يجيب اعتمادًا على المقاطع المسترجعة فقط
```

---

## 🧰 التقنيات المستخدمة

| المكوّن | الأداة |
|---|---|
| تحميل الملف | `PyPDFLoader` |
| تقسيم النص | `RecursiveCharacterTextSplitter` (حجم 500، تداخل 50) |
| Embeddings | `HuggingFaceEmbeddings` بنموذج `all-MiniLM-L6-v2` |
| قاعدة المتجهات | Qdrant (في الذاكرة) |
| البحث بالكلمات | BM25 |
| النموذج اللغوي | Gemini (`gemini-2.5-flash`) |
| الواجهة | Streamlit |
| الإطار | LangChain |

---

## 📁 هيكل المشروع

```
.
├── app.py             # واجهة Streamlit
├── rag.py             # منطق الـ RAG (التحميل، الفهرسة، البحث، Gemini)
├── requirements.txt   # المكتبات المطلوبة
├── .env               # مفتاح Gemini (لا يُرفع على GitHub)
└── README.md
```

---

## ⚙️ التثبيت والتشغيل

### 1) تثبيت المكتبات

```bash
python -m venv venv
venv\Scripts\activate          # على Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2) مفتاح Gemini

احصل على المفتاح من [Google AI Studio](https://aistudio.google.com/apikey)، ثم أنشئ ملف `.env` بجوار `rag.py`:

```
GOOGLE_API_KEY=your_api_key_here
```

> ⚠️ لا ترفع ملف `.env` على GitHub أبدًا. أضفه إلى `.gitignore`.

### 3) تحديد مسار الكتاب

افتح `rag.py` وعدّل المتغيرين التاليين حسب كتابك:

```python
PDF_PATH = r"D:\LLM\AI Agent\project\python-for-data-analysis.pdf"
KNOWLEDGE_BASE = "Python for Data Analysis"   # اسم الكتاب (يُستخدم في الـ prompts)
```

### 4) تشغيل التطبيق

```bash
streamlit run app.py
```

في أول تشغيل يتم تنزيل نموذج الـ embeddings وفهرسة الكتاب، وقد يستغرق ذلك بعض الوقت.

---

## 🖥️ استخدام الواجهة

- اكتب سؤالك في خانة المحادثة (عربي أو إنجليزي).
- من الشريط الجانبي يمكنك:
  - ضبط عدد النتائج لكل استعلام (`k`).
  - إظهار أو إخفاء الاستعلامات المولّدة.
  - تفعيل **فلتر الصفحات** للبحث في نطاق محدد من الكتاب.
  - مسح المحادثة.

مثال على سؤال:

```
إزاي أتعامل مع القيم الفاضية في pandas؟
```

يمكن أيضًا اختبار المحرك من الـ Terminal بدون واجهة:

```bash
python rag.py
```

---

## ☁️ التشغيل على Google Colab

1. ثبّت المكتبات من `requirements.txt`.
2. ارفع `rag.py` و`app.py` و`.env` والكتاب.
3. شغّل `streamlit run app.py` في الخلفية، وافتح التطبيق عبر نفق مثل `cloudflared`.

> على Colab لا يوجد المسار `D:\...`، لذلك عدّل `PDF_PATH` ليشير إلى مكان الكتاب داخل Colab (مثل `/content/book.pdf`).

---

## 🔐 استخدام GitHub Secrets

المفتاح لا يُكتب في الكود، ويُقرأ من متغير البيئة `GOOGLE_API_KEY`.
لاستخدامه في GitHub Actions:

`Settings ← Secrets and variables ← Actions ← New repository secret`

```yaml
env:
  GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

---

## 📝 ملاحظات مهمة

- نموذج `all-MiniLM-L6-v2` **إنجليزي فقط**. إن كان الكتاب عربيًا فغيّره إلى `paraphrase-multilingual-MiniLM-L12-v2`.
- قاعدة Qdrant تعمل في الذاكرة (`:memory:`)، وتُعاد الفهرسة عند كل تشغيل. للحفظ الدائم استخدم `path="./qdrant_db"` بدل `location=":memory:"`.
- ملف الكتاب غير مرفق مع المشروع لأنه محمي بحقوق النشر.
- الخطة المجانية في Gemini لها حدود استخدام؛ كل سؤال يستدعي النموذج 3 مرات (إعادة صياغة، توسيع، إجابة).

---

## 🚀 أفكار للتطوير

- Parent Document Retriever للحفاظ على سياق أوسع.
- Contextual Retrieval لإضافة وصف سياقي لكل chunk.
- Corrective RAG وAdaptive RAG للتحقق من الإجابة والتوجيه بين المصادر.
