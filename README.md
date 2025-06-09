# 🚀 Streamlit Cloud Deployment Guide

Complete guide to deploy your AI-Powered Contact Finder on Streamlit Cloud with zero issues.

## 🔧 **Pre-Deployment Setup**

### **1. Repository Structure**
Ensure your repository has these files:
```
your-repo/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Streamlit Cloud compatible dependencies
├── .streamlit/
│   └── config.toml          # Optional: Streamlit configuration
├── README.md                # Project documentation
└── .gitignore              # Git ignore file
```

### **2. Rename Main File**
Streamlit Cloud looks for specific file names. Rename your main file to one of:
- `streamlit_app.py` (recommended)
- `app.py`
- `main.py`

```bash
# Rename the main application file
mv scrapy_email_finder.py streamlit_app.py
```

### **3. Update Requirements.txt**
Use the Streamlit Cloud compatible version (no Scrapy dependencies):

```txt
streamlit>=1.28.0
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
validators>=0.22.0
python-dotenv>=1.0.0
whois>=0.9.27
openpyxl>=3.1.0
xlsxwriter>=3.1.0
urllib3>=2.0.0
chardet>=5.2.0
lxml>=4.9.0
email-validator>=2.1.0
certifi>=2023.7.22
idna>=3.4
fake-useragent>=1.4.0
```

## 🔐 **API Key Configuration**

### **Method 1: Streamlit Secrets (Recommended)**

1. **In your Streamlit Cloud dashboard**, go to your app settings
2. **Click "Secrets"** in the left sidebar
3. **Add your API key**:
```toml
OPENROUTER_API_KEY = "your_actual_api_key_here"
```

### **Method 2: Environment Variables**
Alternatively, set environment variables in the Streamlit Cloud interface.

### **Method 3: User Input (Fallback)**
The app will ask for API key input if not found in secrets.

## 🚀 **Deployment Steps**

### **1. Push to GitHub**
```bash
git add .
git commit -m "Streamlit Cloud compatible version"
git push origin main
```

### **2. Deploy on Streamlit Cloud**

1. **Visit** [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**
5. **Choose branch**: `main` (or your preferred branch)
6. **Main file path**: `streamlit_app.py`
7. **Click "Deploy!"**

### **3. Configure Secrets**
1. **Go to app settings** (gear icon)
2. **Click "Secrets"** in sidebar
3. **Add your OpenRouter API key**:
```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```
4. **Save** and the app will automatically restart

## ✅ **Verification Checklist**

### **Before Deployment:**
- [ ] Main file named `streamlit_app.py`
- [ ] Compatible `requirements.txt` (no Scrapy)
- [ ] All files committed to GitHub
- [ ] OpenRouter API key ready

### **After Deployment:**
- [ ] App loads without errors
- [ ] API key configured in secrets
- [ ] Model selection works
- [ ] Web scraping functionality works
- [ ] AI research functionality works
- [ ] WHOIS lookup works
- [ ] Batch CSV processing works
- [ ] Export functions work

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Module Import Errors**
```
ModuleNotFoundError: No module named 'scrapy'
```
**Solution**: Use the Streamlit Cloud compatible requirements.txt without Scrapy dependencies.

#### **2. API Key Not Found**
```
API key required to proceed
```
**Solutions**:
- Add API key to Streamlit secrets (recommended)
- Enter API key in the app interface
- Check secrets syntax in Streamlit dashboard

#### **3. Memory Limits**
```
Resource limit exceeded
```
**Solutions**:
- Reduce batch processing size
- Process smaller companies lists
- Use faster/smaller AI models

#### **4. Timeout Issues**
```
Request timeout
```
**Solutions**:
- Check internet connectivity
- Try different AI models
- Reduce request timeout in code

#### **5. WHOIS Lookup Fails**
```
WHOIS lookup failed
```
**Solutions**:
- Some domains may block WHOIS requests
- This is normal for certain TLDs
- Other methods (web scraping, AI) will still work

### **Performance Optimization**

#### **For Better Streamlit Cloud Performance:**

1. **Use Efficient Models**:
   - Free models: For testing and light usage
   - Premium models: For best accuracy
   - Web search models: For comprehensive research

2. **Optimize Batch Processing**:
   - Process 10-20 companies at a time
   - Use progress indicators
   - Add delays between requests

3. **Cache Configuration**:
```python
@st.cache_data
def get_openrouter_models(api_key):
    # Cache model list for better performance
```

## 📊 **Usage Guidelines**

### **Free Tier Limitations**
- **CPU**: Limited processing time
- **Memory**: 1GB RAM limit
- **Network**: Standard bandwidth
- **Concurrent Users**: Limited

### **Recommended Usage**
- **Single Company**: Works perfectly
- **Batch Processing**: 10-50 companies per batch
- **API Calls**: Monitor OpenRouter usage
- **Export**: CSV and Excel work fine

## 🆘 **Support & Resources**

### **Streamlit Cloud Documentation**
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Secrets Management](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Resource Limits](https://docs.streamlit.io/streamlit-cloud/get-started/limitations)

### **OpenRouter API**
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [API Key Management](https://openrouter.ai/keys)
- [Model Pricing](https://openrouter.ai/models)

### **Common Error Solutions**

#### **Deployment Fails**
1. Check requirements.txt syntax
2. Ensure all imports are available
3. Remove Scrapy-related dependencies
4. Check GitHub repository access

#### **App Crashes on Load**
1. Check Streamlit Cloud logs
2. Verify API key in secrets
3. Test locally first
4. Check for missing dependencies

#### **Slow Performance**
1. Use smaller batch sizes
2. Choose faster AI models
3. Optimize web scraping requests
4. Consider upgrading Streamlit plan

## 🎯 **Best Practices**

### **Code Organization**
```python
# Use session state for persistence
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Handle API keys gracefully
api_key = st.secrets.get("OPENROUTER_API_KEY") or st.text_input("API Key", type="password")

# Add error handling
try:
    result = process_company(...)
except Exception as e:
    st.error(f"Error: {e}")
```

### **User Experience**
- Clear progress indicators
- Helpful error messages
- Responsive design
- Mobile-friendly interface

### **Resource Management**
- Cache expensive operations
- Limit concurrent requests
- Handle rate limiting gracefully
- Provide download options

## 📈 **Scaling Considerations**

### **For Heavy Usage**
- Consider Streamlit Cloud Pro
- Implement request queuing
- Add user authentication
- Monitor API usage costs

### **Alternative Deployment**
If Streamlit Cloud limits are restrictive:
- Deploy on Heroku
- Use DigitalOcean Apps
- Deploy on Google Cloud Run
- Self-host on VPS

---

**🎉 Your AI-Powered Contact Finder is now ready for Streamlit Cloud!**

**Next Steps:**
1. Deploy using the guide above
2. Test all functionality
3. Share your app URL
4. Monitor usage and performance
