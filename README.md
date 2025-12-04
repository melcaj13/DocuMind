# 🧠 DocuMind: Akıllı Doküman Asistanı (RAG)

DocuMind, kullanıcıların PDF dokümanlarıyla sohbet etmesine olanak tanıyan bir anlamsal arama ve soru-cevap sistemidir. **Retrieval-Augmented Generation (RAG)** mimarisini kullanarak, sorulara *sadece* verilen doküman bağlamında yanıt verir ve yapay zeka halüsinasyonlarını (uydurma cevapları) engeller.

## 📺 Proje Demosu





https://github.com/user-attachments/assets/0a6a99f5-683c-4764-8858-4f279de969bf





## 🚀 Özellikler

* **Anlamsal Arama (Semantic Search):** Sadece kelime eşleşmesine değil, sorunun arkasındaki niyete ve anlama odaklanır.
* **Halüsinasyon Kontrolü:** Eğer cevap dokümanda yoksa, yapay zeka bu bilgiyi bulamadığını açıkça belirtir. Asla uydurmaz.
* **Kaynak Şeffaflığı:** Kullanıcılar, cevabın üretildiği orijinal metin parçasını görüntüleyebilir.
* **Kullanıcı Dostu Arayüz:** Sürükle-bırak özelliği ve görsel geri bildirimler içeren temiz bir arayüz.

## 🛠️ Teknoloji Yığını (Tech Stack)

* **Dil:** Python 
* **Arayüz:** Streamlit
* **Orkestrasyon:** LangChain
* **Yapay Zeka Modelleri:** HuggingFace / Sentence Transformers
* **Vektör Veritabanı:** FAISS (Facebook AI Similarity Search)
* **Veri İşleme:** PDF işleme için `pypdf`
  
## ⚠️ Language Support / Dil Desteği

* Bu projede kullanılan mevcut API/LLM modeli **İngilizce** için optimize edilmiştir. Sistem diğer dilleri işleyebilse de, en doğru ve hatasız sonuçlar İngilizce dokümanlar ve sorgular kullanıldığında elde edilir.

## 🎯 Nasıl Çalışır?

1.  **Yükleme:** Kullanıcı sisteme bir PDF dosyası yükler.
2.  **Gömme (Embedding):** Metin parçalara ayrılır ve `SentenceTransformers` kullanılarak vektör sayılarına dönüştürülür.
3.  **Geri Getirme (Retrieval):** Kullanıcı bir soru sorduğunda, sistem **FAISS** veritabanında en alakalı parçaları bulur.
4.  **Üretim (Generation):** LLM, *sadece* getirilen bu bağlamı kullanarak cevabı oluşturur.
