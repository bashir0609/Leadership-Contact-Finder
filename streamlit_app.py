import streamlit as st
import requests
import pandas as pd
import re
from datetime import datetime
import validators
from urllib.parse import urlparse, urljoin
import time
import json
import io
from concurrent.futures import ThreadPoolExecutor
import threading
from bs4 import BeautifulSoup
import urllib.robotparser
import subprocess

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

class WebScraper:
    """Simple web scraper for extracting contact information from websites"""
    
    def __init__(self, company_name=""):
        self.company_name = company_name
        self.contacts_found = {
            'emails': set(),
            'phones': set(),
            'names': set(),
            'addresses': set(),
            'social_links': set(),
            'pages_scraped': [],
            'contact_pages': []
        }
        
        # Email and phone regex patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?\d{1,4}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        )
        
        # Contact page keywords
        self.contact_keywords = [
            'contact', 'kontakt', 'contacts', 'contact-us', 'contact_us',
            'about', 'about-us', 'about_us', 'team', 'staff', 'people',
            'impressum', 'imprint', 'mentions-legales', 'legal',
            'leadership', 'management', 'executives', 'directors',
            'office', 'offices', 'locations', 'address', 'phone'
        ]
        
        # Session for maintaining connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ContactFinder/1.0 (+https://contact-finder.app)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def can_fetch(self, url):
        """Check if we can fetch the URL according to robots.txt"""
        try:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(url, '/robots.txt')
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch('*', url)
        except Exception:
            # If we can't check robots.txt, assume it's okay but be respectful
            return True
    
    def fetch_page(self, url, timeout=30):
        """Fetch a single page with error handling"""
        try:
            if not self.can_fetch(url):
                return None
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            # Don't show 404 errors as warnings - they're expected
            if e.response.status_code == 404:
                return None
            else:
                st.warning(f"HTTP error {e.response.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            # Only show significant errors
            if "timeout" in str(e).lower():
                st.warning(f"Timeout fetching {url}")
            return None
    
    def extract_contacts_from_text(self, text, url=""):
        """Extract contact information from text"""
        # Extract emails
        emails = self.email_pattern.findall(text)
        for email in emails:
            # Filter out common non-contact emails
            if not any(skip in email.lower() for skip in 
                      ['noreply', 'no-reply', 'donotreply', 'example.com', 'test.com']):
                self.contacts_found['emails'].add(email.lower())
        
        # Extract phone numbers
        phones = self.phone_pattern.findall(text)
        for phone in phones:
            # Clean and validate phone numbers
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) >= 7:  # Minimum phone length
                self.contacts_found['phones'].add(phone.strip())
    
    def extract_names_from_soup(self, soup):
        """Extract potential contact names from BeautifulSoup object"""
        # Common name contexts
        name_contexts = [
            'team', 'staff', 'contact', 'management', 'leadership',
            'ceo', 'cto', 'cfo', 'director', 'manager', 'head'
        ]
        
        # Find elements that might contain names
        for context in name_contexts:
            elements = soup.find_all(text=re.compile(context, re.I))
            for element in elements:
                parent = element.parent if element.parent else element
                # Extract potential names from nearby text
                text = parent.get_text() if hasattr(parent, 'get_text') else str(parent)
                names = self.extract_person_names(text)
                self.contacts_found['names'].update(names)
    
    def extract_person_names(self, text):
        """Extract person names using pattern matching"""
        names = set()
        
        # Pattern for names (First Last, Dr. First Last, etc.)
        name_pattern = r'\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(name_pattern, text)
        
        for match in matches:
            # Filter out common non-names
            if not any(word.lower() in match.lower() for word in 
                      ['contact', 'email', 'phone', 'address', 'website', 'company']):
                if len(match.split()) >= 2:  # At least first and last name
                    names.add(match.strip())
        
        return names
    
    def extract_addresses_from_text(self, text):
        """Extract physical addresses"""
        # Address patterns (simplified)
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[\s,]+[A-Za-z\s]+,?\s*\d{5}',
            r'[A-Za-z\s]+\d+\s*,\s*\d{5}\s+[A-Za-z\s]+',  # European format
        ]
        
        for pattern in address_patterns:
            addresses = re.findall(pattern, text, re.IGNORECASE)
            for address in addresses:
                self.contacts_found['addresses'].add(address.strip())
    
    def extract_social_links(self, soup):
        """Extract social media links"""
        social_domains = [
            'linkedin.com', 'xing.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'youtube.com', 'tiktok.com'
        ]
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            for domain in social_domains:
                if domain in href:
                    self.contacts_found['social_links'].add(href)
    
    def find_contact_links(self, soup, base_url):
        """Find links that likely lead to contact pages"""
        contact_links = []
        links = soup.find_all('a', href=True)
        
        # Look for contact-related links in the actual page content
        for link in links:
            href = link['href']
            link_text = (link.get_text() or '').lower()
            href_lower = href.lower()
            
            # Check both href and link text for contact keywords
            is_contact_link = (
                any(keyword in href_lower for keyword in self.contact_keywords) or
                any(keyword in link_text for keyword in ['contact', 'about', 'team', 'impressum'])
            )
            
            if is_contact_link:
                absolute_url = urljoin(base_url, href)
                contact_links.append(absolute_url)
        
        return list(set(contact_links))  # Remove duplicates
    
    def scrape_website(self, start_url, max_pages=5):
        """Scrape website for contact information"""
        try:
            # Ensure URL has protocol
            if not start_url.startswith(('http://', 'https://')):
                start_url = 'https://' + start_url
            
            domain = urlparse(start_url).netloc
            visited_pages = set()
            pages_found = 0
            
            progress_placeholder = st.empty()
            
            # Step 1: Scrape homepage first
            progress_placeholder.info(f"🏠 Analyzing homepage: {start_url}")
            
            homepage_response = self.fetch_page(start_url)
            if not homepage_response:
                st.warning(f"Could not access homepage: {start_url}")
                return None
            
            visited_pages.add(start_url)
            self.contacts_found['pages_scraped'].append(start_url)
            pages_found += 1
            
            # Parse homepage
            homepage_soup = BeautifulSoup(homepage_response.content, 'html.parser')
            homepage_text = homepage_soup.get_text()
            
            # Extract contacts from homepage
            self.extract_contacts_from_text(homepage_text, start_url)
            self.extract_names_from_soup(homepage_soup)
            self.extract_addresses_from_text(homepage_text)
            self.extract_social_links(homepage_soup)
            
            # Step 2: Find actual contact links from homepage
            contact_links_found = self.find_contact_links(homepage_soup, start_url)
            
            progress_placeholder.info(f"🔍 Found {len(contact_links_found)} potential contact pages")
            
            # Step 3: Try contact links found on homepage
            for contact_url in contact_links_found[:max_pages-1]:  # Reserve space for homepage
                if contact_url in visited_pages or pages_found >= max_pages:
                    continue
                
                progress_placeholder.info(f"📄 Scraping contact page {pages_found+1}: {urlparse(contact_url).path}")
                
                response = self.fetch_page(contact_url)
                if not response:
                    continue
                
                visited_pages.add(contact_url)
                self.contacts_found['pages_scraped'].append(contact_url)
                self.contacts_found['contact_pages'].append(contact_url)
                pages_found += 1
                
                # Parse contact page
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                
                # Extract contacts with higher confidence from contact pages
                self.extract_contacts_from_text(text, contact_url)
                self.extract_names_from_soup(soup)
                self.extract_addresses_from_text(text)
                self.extract_social_links(soup)
                
                # Add delay to be respectful
                time.sleep(1)
            
            # Step 4: If we haven't found many pages, try common paths as fallback
            if pages_found < 3:
                common_paths = ['/contact', '/about', '/team', '/kontakt', '/impressum']
                
                for path in common_paths:
                    if pages_found >= max_pages:
                        break
                        
                    fallback_url = urljoin(start_url, path)
                    if fallback_url in visited_pages:
                        continue
                    
                    response = self.fetch_page(fallback_url)
                    if not response:
                        continue  # 404s are normal, don't show as errors
                    
                    visited_pages.add(fallback_url)
                    self.contacts_found['pages_scraped'].append(fallback_url)
                    self.contacts_found['contact_pages'].append(fallback_url)
                    pages_found += 1
                    
                    # Parse page
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    
                    self.extract_contacts_from_text(text, fallback_url)
                    self.extract_names_from_soup(soup)
                    self.extract_addresses_from_text(text)
                    self.extract_social_links(soup)
                    
                    time.sleep(1)
            
            progress_placeholder.success(f"✅ Web scraping completed! Analyzed {pages_found} pages")
            
            # Convert sets to lists for JSON serialization
            return {
                'emails': list(self.contacts_found['emails']),
                'phones': list(self.contacts_found['phones']),
                'names': list(self.contacts_found['names']),
                'addresses': list(self.contacts_found['addresses']),
                'social_links': list(self.contacts_found['social_links']),
                'pages_scraped': self.contacts_found['pages_scraped'],
                'contact_pages': self.contacts_found['contact_pages']
            }
            
        except Exception as e:
            st.error(f"Web scraping failed: {e}")
            return None

def get_openrouter_models(api_key):
    """Get comprehensive list of OpenRouter models with categories"""
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", 
                           headers={"Authorization": f"Bearer {api_key}"})
        models = resp.json()["data"]
        
        # Categorize models
        free_models = []
        web_search_models = []
        premium_models = []
        
        for model in models:
            model_id = model["id"]
            model_name = model.get("name", model_id)
            pricing = model.get("pricing", {})
            
            # Check if free (some models have 0 cost)
            is_free = (pricing.get("prompt", "0") == "0" and 
                      pricing.get("completion", "0") == "0")
            
            # Web search capable models
            if any(keyword in model_id.lower() for keyword in 
                   ["perplexity", "online", "web", "search", "sonar"]):
                web_search_models.append((model_id, f"{model_name} (Web Search)"))
            elif is_free:
                free_models.append((model_id, f"{model_name} (Free)"))
            else:
                premium_models.append((model_id, f"{model_name}"))
        
        return {
            "web_search": sorted(web_search_models),
            "free": sorted(free_models),
            "premium": sorted(premium_models)
        }
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return {"web_search": [], "free": [], "premium": []}

def create_comprehensive_search_prompt(company, website, country, industry=""):
    """Enhanced prompt for comprehensive contact finding"""
    domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    
    prompt = f"""
You are an expert business intelligence researcher with access to real-time web data. Your task is to find comprehensive contact information for {company} (website: {website}) in {country}.

**COMPREHENSIVE SEARCH STRATEGY - Execute ALL sources:**

1. **Official Website Deep Analysis**:
   - {website}/contact, /about, /team, /staff, /leadership
   - {website}/impressum (German sites), /mentions-legales (French)
   - Subdirectories: /en/contact, /de/kontakt
   - Extract ALL emails, names, phone numbers, addresses

2. **LinkedIn Professional Search**:
   - Current employees at "{company}"
   - Search variations: "{company} GmbH", "{company} Ltd", etc.
   - Executives: CEO, CTO, CFO, VP, Director, Manager
   - Department heads: HR, Sales, Marketing, Operations
   - Get LinkedIn profile URLs and contact information

3. **Business Directory Mining**:
   - Google Business listings
   - Yelp, Yellow Pages, local directories
   - Industry-specific directories
   - Chamber of Commerce listings
   - Better Business Bureau (US)
   - Companies House (UK), Bundesanzeiger (Germany)

4. **Professional Networks**:
   - Xing.com (German-speaking countries)
   - AngelList (startups)
   - Crunchbase (funding/executive info)
   - Industry association member directories

5. **News & Press Coverage**:
   - Recent press releases mentioning executives
   - Industry publications and interviews
   - Conference speaker lists
   - Award announcements
   - Local news mentions

6. **Technical Sources**:
   - WHOIS domain registration data
   - SSL certificate contact info
   - DNS records and mail server information
   - Cached versions (Wayback Machine)

7. **Social Media & Web Presence**:
   - Company Facebook, Twitter, Instagram pages
   - YouTube channel information
   - Medium, blog author information
   - Podcast guest appearances

8. **Government & Legal Sources**:
   - Corporate registration databases
   - SEC filings (public companies)
   - Patent applications
   - Trademark registrations
   - Court filings

**OUTPUT FORMAT** (Markdown Table):

| Name | Role/Title | LinkedIn/Profile URL | Email | Phone | Source | Confidence | Notes |
|------|------------|---------------------|--------|-------|--------|------------|-------|
| [Name or "General Contact"] | [Exact Title] | [Full URL] | [Email] | [Phone] | [Source] | [High/Med/Low] | [Additional info] |

**EMAIL PATTERNS TO CHECK**:
- info@{domain}, contact@{domain}, hello@{domain}
- sales@, support@, office@, admin@
- firstname@, f.lastname@, firstname.lastname@
- Common patterns for this country/industry

**CONFIDENCE LEVELS**:
- **High**: Verified from official sources (website, LinkedIn, press)
- **Medium**: Business directories, news articles
- **Low**: Pattern-based estimates, unverified sources

**CRITICAL**: Search EVERY source mentioned above. Don't skip any category.

Begin comprehensive research for {company} now. Be thorough and systematic.
"""
    
    return prompt

def check_gemini_availability():
    """Check if Google GenerativeAI library is available"""
    try:
        import google.generativeai as genai
        return True
    except ImportError:
        return False

def query_gemini_api(api_key, prompt, timeout=120):
    """Query Google Gemini API for contact research"""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4000,
            )
        )
        
        return response.text
        
    except ImportError:
        st.error("🚨 Google GenerativeAI library not available. Please check requirements.txt installation.")
        st.info("💡 **Fix**: Add `google-generativeai>=0.7.0` to requirements.txt and redeploy")
        return None
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

def get_available_ai_providers(openrouter_key=None, gemini_key=None):
    """Get list of available AI providers and their models"""
    providers = {}
    
    # Check Gemini availability
    gemini_available = check_gemini_availability()
    
    # OpenRouter models
    if openrouter_key:
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", 
                               headers={"Authorization": f"Bearer {openrouter_key}"})
            models = resp.json()["data"]
            
            # Categorize OpenRouter models
            free_models = []
            web_search_models = []
            premium_models = []
            
            for model in models:
                model_id = model["id"]
                model_name = model.get("name", model_id)
                pricing = model.get("pricing", {})
                
                is_free = (pricing.get("prompt", "0") == "0" and 
                          pricing.get("completion", "0") == "0")
                
                if any(keyword in model_id.lower() for keyword in 
                       ["perplexity", "online", "web", "search", "sonar"]):
                    web_search_models.append((model_id, f"{model_name} (Web Search)"))
                elif is_free:
                    free_models.append((model_id, f"{model_name} (Free)"))
                else:
                    premium_models.append((model_id, f"{model_name}"))
            
            providers["openrouter"] = {
                "name": "OpenRouter",
                "models": {
                    "web_search": sorted(web_search_models),
                    "free": sorted(free_models),
                    "premium": sorted(premium_models)
                }
            }
        except:
            pass
    
    # Gemini models (only if library is available and API key provided)
    if gemini_key and gemini_available:
        providers["gemini"] = {
            "name": "Google Gemini",
            "models": {
                "gemini": [
                    ("gemini-1.5-pro-latest", "Gemini 1.5 Pro (Latest)"),
                    ("gemini-1.5-pro", "Gemini 1.5 Pro"),
                    ("gemini-1.5-flash", "Gemini 1.5 Flash (Faster)"),
                    ("gemini-pro", "Gemini Pro")
                ]
            }
        }
    elif gemini_key and not gemini_available:
        # Show warning if API key provided but library not available
        st.warning("⚠️ Gemini API key provided but google-generativeai library not installed")
    
    return providers

def query_ai_enhanced(provider, api_key, model, prompt, timeout=120):
    """Enhanced AI query supporting multiple providers"""
    if provider == "openrouter":
        return query_openrouter_enhanced(api_key, model, prompt, timeout)
    elif provider == "gemini":
        return query_gemini_api(api_key, prompt, timeout)
    else:
        st.error(f"Unknown AI provider: {provider}")
        return None
def query_openrouter_enhanced(api_key, model, prompt, timeout=120):
    """Enhanced OpenRouter API query with better error handling"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-contact-finder.streamlit.app",
        "X-Title": "AI Contact Finder"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9
    }
    
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 15
                st.warning(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 402:
                st.error("Insufficient credits. Please check your OpenRouter account.")
                return None
            else:
                st.error(f"API error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timeout on attempt {attempt + 1}/3")
            if attempt < 2:
                time.sleep(10)
        except Exception as e:
            if attempt == 2:
                st.error(f"Failed after 3 attempts: {str(e)}")
                return None
            time.sleep(5)
    
    return None

def get_whois_contacts(domain):
    """Extract comprehensive contacts from WHOIS data"""
    try:
        # Try different whois implementations
        w = None
        
        # Method 1: Try python-whois
        try:
            import whois
            if hasattr(whois, 'whois'):
                w = whois.whois(domain)
            elif hasattr(whois, 'query'):
                w = whois.query(domain)
        except:
            pass
        
        # Method 2: Try alternative whois approach
        if w is None:
            try:
                import subprocess
                import json
                result = subprocess.run(['whois', domain], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    # Parse basic info from whois text output
                    whois_text = result.stdout.lower()
                    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', result.stdout)
                    
                    return {
                        'domain': domain,
                        'registrar': None,
                        'creation_date': None,
                        'expiration_date': None,
                        'name_servers': None,
                        'emails': list(set(emails)) if emails else [],
                        'org': None,
                        'country': None,
                        'raw_data': result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
                    }
            except:
                pass
        
        # Method 3: If we have whois object, parse it
        if w:
            contacts = {
                'domain': domain,
                'registrar': getattr(w, 'registrar', None),
                'creation_date': getattr(w, 'creation_date', None),
                'expiration_date': getattr(w, 'expiration_date', None),
                'name_servers': getattr(w, 'name_servers', None),
                'emails': [],
                'org': getattr(w, 'org', None),
                'country': getattr(w, 'country', None)
            }
            
            # Extract all emails
            if hasattr(w, 'emails') and w.emails:
                if isinstance(w.emails, list):
                    contacts['emails'] = list(set(w.emails))  # Remove duplicates
                else:
                    contacts['emails'] = [w.emails]
            
            # Get additional fields
            for field in ['admin_email', 'tech_email', 'billing_email']:
                if hasattr(w, field):
                    email = getattr(w, field)
                    if email and email not in contacts['emails']:
                        contacts['emails'].append(email)
            
            return contacts
        
        # If all methods fail, return minimal info
        return {
            'domain': domain,
            'registrar': None,
            'creation_date': None,
            'expiration_date': None,
            'name_servers': None,
            'emails': [],
            'org': None,
            'country': None,
            'error': 'WHOIS lookup not available'
        }
            
    except Exception as e:
        # Return error info instead of None
        return {
            'domain': domain,
            'registrar': None,
            'creation_date': None,
            'expiration_date': None,
            'name_servers': None,
            'emails': [],
            'org': None,
            'country': None,
            'error': f'WHOIS lookup failed: {str(e)}'
        }

def process_single_company(provider, api_key, model, company, website, country, industry="", search_methods=None):
    """Process a single company and return results"""
    try:
        # Validate website
        if not website.startswith(('http://', 'https://')):
            website = 'https://' + website
        
        if not validators.url(website):
            return {
                'company': company,
                'website': website,
                'error': 'Invalid website URL'
            }
        
        domain = urlparse(website).netloc.replace('www.', '')
        
        results = {
            'company': company,
            'website': website,
            'domain': domain,
            'country': country,
            'industry': industry,
            'whois_data': None,
            'ai_research': None,
            'web_scraping_data': None,
            'ai_provider': provider,
            'ai_model': model,
            'processed_at': datetime.now()
        }
        
        # WHOIS lookup
        if "WHOIS Lookup" in search_methods:
            results['whois_data'] = get_whois_contacts(domain)
        
        # Web scraping
        if "Website Scraping" in search_methods:
            scraper = WebScraper(company)
            results['web_scraping_data'] = scraper.scrape_website(website)
        
        # AI research
        if "AI Research" in search_methods:
            prompt = create_comprehensive_search_prompt(company, website, country, industry)
            results['ai_research'] = query_ai_enhanced(provider, api_key, model, prompt)
        
        return results
        
    except Exception as e:
        return {
            'company': company,
            'website': website,
            'error': str(e)
        }

def display_web_scraping_results(scraping_data):
    """Display web scraping results"""
    if not scraping_data:
        st.warning("No web scraping data available")
        return
    
    with st.expander("🌐 Website Scraping Results", expanded=True):
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emails Found", len(scraping_data.get('emails', [])))
        with col2:
            st.metric("Phone Numbers", len(scraping_data.get('phones', [])))
        with col3:
            st.metric("Names Extracted", len(scraping_data.get('names', [])))
        with col4:
            st.metric("Pages Scraped", len(scraping_data.get('pages_scraped', [])))
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📧 Email Addresses")
            emails = scraping_data.get('emails', [])
            if emails:
                for email in emails:
                    st.code(email)
            else:
                st.info("No emails found")
            
            st.subheader("📞 Phone Numbers")
            phones = scraping_data.get('phones', [])
            if phones:
                for phone in phones:
                    st.code(phone)
            else:
                st.info("No phone numbers found")
        
        with col2:
            st.subheader("👥 Names Extracted")
            names = scraping_data.get('names', [])
            if names:
                for name in names:
                    st.code(name)
            else:
                st.info("No names found")
            
            st.subheader("🏢 Addresses")
            addresses = scraping_data.get('addresses', [])
            if addresses:
                for address in addresses:
                    st.code(address)
            else:
                st.info("No addresses found")
        
        # Social links
        if scraping_data.get('social_links'):
            st.subheader("🔗 Social Media Links")
            for link in scraping_data.get('social_links', []):
                st.markdown(f"- [{link}]({link})")
        
        # Pages scraped
        if scraping_data.get('pages_scraped'):
            st.subheader("📄 Pages Scraped")
            for page in scraping_data.get('pages_scraped', []):
                st.markdown(f"- [{page}]({page})")

def parse_ai_results_to_dataframe(ai_result):
    """Parse AI research results into a structured dataframe"""
    if not ai_result:
        return None
    
    try:
        lines = ai_result.split("\n")
        table_lines = [line for line in lines if "|" in line and "---" not in line and "Name" in line or 
                      ("|" in line and "---" not in line and len([cell for cell in line.split("|") if cell.strip()]) >= 4)]
        
        if len(table_lines) >= 2:
            # Extract headers
            headers = [cell.strip() for cell in table_lines[0].split("|") if cell.strip()]
            
            # Extract data rows
            rows = []
            for line in table_lines[1:]:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if len(cells) >= 3:  # Minimum viable row
                    # Pad with empty strings if needed
                    while len(cells) < len(headers):
                        cells.append("")
                    rows.append(cells[:len(headers)])
            
            if rows:
                df = pd.DataFrame(rows, columns=headers)
                return df
    except Exception as e:
        st.warning(f"Could not parse table format: {e}")
    
    return None

def main():
    st.set_page_config(
        page_title="AI-Powered Contact Finder",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI-Powered Contact Finder")
    st.markdown("*Web scraping + Multi-AI research + WHOIS lookup + Professional networks + Batch processing*")
    st.markdown("**✅ Supports OpenRouter (100+ models) + Google Gemini (latest AI)**")
    
    # Load API keys from Streamlit secrets or user input
    st.subheader("🔐 AI Provider Configuration")
    
    # Check for API keys in secrets
    openrouter_key = None
    gemini_key = None
    
    try:
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
        gemini_key = st.secrets.get("GEMINI_API_KEY")
    except:
        pass
    
    # Create columns for API key inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**OpenRouter API**")
        if not openrouter_key:
            openrouter_key = st.text_input(
                "OpenRouter API Key", 
                type="password",
                help="Get your API key from https://openrouter.ai/",
                key="openrouter_input"
            )
        else:
            st.success("✅ OpenRouter API key loaded from secrets")
    
    with col2:
        st.markdown("**Google Gemini API**")
        
        # Check if Gemini library is available
        gemini_available = check_gemini_availability()
        
        if not gemini_available:
            st.warning("⚠️ Google GenerativeAI library not installed")
            with st.expander("🔧 Quick Fix Instructions"):
                st.markdown("""
                **Add to requirements.txt:**
                ```
                google-generativeai>=0.7.0
                ```
                
                **Then redeploy your app.**
                
                **Alternative:** Use OpenRouter only - it has 100+ models including web search!
                """)
        
        if not gemini_key:
            gemini_key = st.text_input(
                "Gemini API Key", 
                type="password",
                help="Get your API key from https://aistudio.google.com/",
                key="gemini_input",
                disabled=not gemini_available
            )
            if not gemini_available:
                st.caption("⚠️ Disabled: Library not installed")
        else:
            if gemini_available:
                st.success("✅ Gemini API key loaded from secrets")
            else:
                st.warning("⚠️ API key available but library not installed")
    
    # Check if at least one API key is available
    if not openrouter_key and not gemini_key:
        st.error("At least one API key is required to proceed")
        st.info("💡 You can add API keys to Streamlit secrets or enter them above")
        st.markdown("""
        **Get API Keys:**
        - **OpenRouter**: [openrouter.ai](https://openrouter.ai/) (Access to 100+ models)
        - **Google Gemini**: [aistudio.google.com](https://aistudio.google.com/) (Free tier available)
        """)
        st.stop()
    
    # Get available providers and models
    with st.spinner("Loading AI providers and models..."):
        providers = get_available_ai_providers(openrouter_key, gemini_key)
    
    # Show provider status
    if providers:
        provider_count = len(providers)
        provider_names = [providers[p]["name"] for p in providers.keys()]
        st.success(f"✅ {provider_count} AI provider(s) available: {', '.join(provider_names)}")
    else:
        st.error("❌ No AI providers available")
        
        # Show specific issues
        if openrouter_key and not any("openrouter" in p for p in providers.keys()):
            st.error("🔑 OpenRouter API key may be invalid")
        
        if gemini_key and not any("gemini" in p for p in providers.keys()):
            if not check_gemini_availability():
                st.error("📦 Google GenerativeAI library not installed")
                st.code("pip install google-generativeai>=0.7.0")
            else:
                st.error("🔑 Gemini API key may be invalid")
        
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Provider and Model selection with persistence
        st.subheader("🤖 AI Provider & Model Selection")
        
        # Provider selection
        provider_names = list(providers.keys())
        provider_options = [providers[p]["name"] for p in provider_names]
        
        if 'selected_provider' not in st.session_state:
            st.session_state.selected_provider = provider_names[0] if provider_names else None
        
        selected_provider_idx = st.selectbox(
            "AI Provider",
            range(len(provider_options)),
            index=provider_names.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_names else 0,
            format_func=lambda x: provider_options[x],
            help="Choose between OpenRouter (100+ models) or Google Gemini (latest models)"
        )
        
        selected_provider = provider_names[selected_provider_idx]
        st.session_state.selected_provider = selected_provider
        
        # Model category selection (for OpenRouter)
        if selected_provider == "openrouter":
            model_category = st.selectbox(
                "Model Category",
                ["Web Search Models (Recommended)", "Free Models", "Premium Models"],
                help="Web Search models can access real-time internet data"
            )
            
            if model_category == "Web Search Models (Recommended)":
                available_models = providers["openrouter"]["models"]["web_search"]
            elif model_category == "Free Models":
                available_models = providers["openrouter"]["models"]["free"]
            else:
                available_models = providers["openrouter"]["models"]["premium"]
        else:
            # Gemini models
            available_models = providers["gemini"]["models"]["gemini"]
        
        if available_models:
            model_options = [f"{name}" for _, name in available_models]
            model_ids = [model_id for model_id, _ in available_models]
            
            # Use session state to persist selection
            session_key = f'selected_model_{selected_provider}'
            if session_key not in st.session_state:
                st.session_state[session_key] = None
            
            if st.session_state[session_key] and st.session_state[session_key] in model_ids:
                default_idx = model_ids.index(st.session_state[session_key])
            else:
                default_idx = 0
            
            selected_idx = st.selectbox(
                "Select Model",
                range(len(model_options)),
                index=default_idx,
                format_func=lambda x: model_options[x]
            )
            
            selected_model = model_ids[selected_idx]
            st.session_state[session_key] = selected_model
        else:
            st.error(f"No models available for {providers[selected_provider]['name']}")
            st.stop()
        
        # Show provider info
        if selected_provider == "openrouter":
            st.info("🌐 **OpenRouter**: Access to 100+ AI models including Perplexity web search")
        elif selected_provider == "gemini":
            st.info("🧠 **Google Gemini**: Latest Google AI with excellent reasoning capabilities")
        
        # Search settings
        st.subheader("🔍 Search Methods")
        
        # Check if WHOIS is available
        whois_available = True
        try:
            import whois
            if not (hasattr(whois, 'whois') or hasattr(whois, 'query')):
                whois_available = False
        except ImportError:
            whois_available = False
        
        default_methods = ["Website Scraping", "AI Research"]
        if whois_available:
            default_methods.append("WHOIS Lookup")
        
        available_methods = ["Website Scraping", "AI Research"]
        if whois_available:
            available_methods.append("WHOIS Lookup")
        else:
            st.warning("⚠️ WHOIS lookup not available in this environment")
        
        search_methods = st.multiselect(
            "Active Search Methods",
            available_methods,
            default=default_methods,
            help="All methods work in Streamlit Cloud"
        )
        
        # Processing mode
        st.subheader("📊 Processing Mode")
        processing_mode = st.radio(
            "Choose Mode",
            ["Single Company", "Batch CSV Processing"]
        )
        
        st.markdown("---")
        st.markdown("**💡 Pro Tips:**")
        st.caption("• Website scraping extracts real contact data")
        st.caption("• Use Web Search models for comprehensive AI research")
        st.caption("• Combine all methods for maximum coverage")
        st.caption("• ✅ All features work in Streamlit Cloud")
    
    # Main content area
    if processing_mode == "Single Company":
        # Single company processing
        st.subheader("🏢 Single Company Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input("Company Name", 
                                  placeholder="e.g., BBW Berufsbildungswerk Hamburg")
            website = st.text_input("Website URL", 
                                  placeholder="e.g., bbw.de or https://bbw.de")
        
        with col2:
            country = st.text_input("Country", value="Germany")
            industry = st.text_input("Industry (Optional)", 
                                   placeholder="e.g., Education, Technology")
        
        if st.button("🚀 Start Multi-Method Search", type="primary"):
            if not all([company, website, country]):
                st.error("Please fill in company name, website, and country")
                return
            
            if not search_methods:
                st.error("Please select at least one search method")
                return
            
            # Process single company
            with st.spinner("Conducting comprehensive multi-method research..."):
                # Get the appropriate API key
                current_api_key = openrouter_key if selected_provider == "openrouter" else gemini_key
                
                result = process_single_company(
                    selected_provider, current_api_key, selected_model, 
                    company, website, country, industry, search_methods
                )
            
            # Display results
            display_single_result(result, search_methods)
    
    else:
        # Batch CSV processing
        st.subheader("📊 Batch CSV Processing")
        
        # CSV template download
        template_df = pd.DataFrame({
            'company': ['Example Corp', 'Müller GmbH', 'José & Associates'],
            'website': ['example.com', 'mueller-company.de', 'jose-associates.com'],
            'country': ['Germany', 'Germany', 'Spain'],
            'industry': ['Technology', 'Manufacturing', 'Consulting']
        })
        
        # Create UTF-8 encoded CSV
        csv_template = template_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            "📥 Download CSV Template (UTF-8)",
            data=csv_template.encode('utf-8'),
            file_name="ai_contact_finder_template.csv",
            mime="text/csv",
            help="Template includes international characters to test encoding"
        )
        
        st.info("💡 **Important**: Save your CSV as UTF-8 format to avoid encoding errors with special characters (ä, ö, ü, é, etc.)")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with companies",
            type=['csv'],
            help="CSV should have columns: company, website, country, industry (optional)"
        )
        
        if uploaded_file:
            try:
                # Try to read CSV with different encodings
                companies_df = None
                encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'windows-1252', 'cp1252']
                
                for encoding in encodings_to_try:
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        companies_df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ CSV loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        if encoding == encodings_to_try[-1]:  # Last encoding attempt
                            raise e
                        continue
                
                if companies_df is None:
                    st.error("Could not read CSV file with any supported encoding. Please save as UTF-8.")
                    st.info("💡 **Tip**: In Excel, use 'Save As' → 'CSV UTF-8 (Comma delimited)' format")
                    return
                
                # Validate required columns
                required_cols = ['company', 'website', 'country']
                missing_cols = [col for col in required_cols if col not in companies_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.info("Required columns: company, website, country, industry (optional)")
                    return
                
                # Clean up data
                companies_df = companies_df.dropna(subset=['company', 'website'])  # Remove rows with missing required data
                companies_df['company'] = companies_df['company'].astype(str).str.strip()
                companies_df['website'] = companies_df['website'].astype(str).str.strip()
                companies_df['country'] = companies_df['country'].astype(str).str.strip()
                
                if len(companies_df) == 0:
                    st.error("No valid companies found in CSV after cleaning")
                    return
                
                st.success(f"✅ Loaded {len(companies_df)} companies")
                st.dataframe(companies_df.head(10))  # Show first 10 rows
                
                if len(companies_df) > 50:
                    st.warning(f"⚠️ Large batch detected ({len(companies_df)} companies). Consider processing in smaller chunks for better performance.")
                
                if st.button("🚀 Process All Companies", type="primary"):
                    # Get the appropriate API key
                    current_api_key = openrouter_key if selected_provider == "openrouter" else gemini_key
                    process_batch_csv(companies_df, selected_provider, current_api_key, selected_model, search_methods)
                    
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                
                # Provide helpful troubleshooting info
                with st.expander("🔧 CSV Troubleshooting Tips"):
                    st.markdown("""
                    **Common CSV Issues & Solutions:**
                    
                    **Character Encoding Problems:**
                    - Save your CSV as "UTF-8" format in Excel
                    - In Excel: File → Save As → CSV UTF-8 (Comma delimited)
                    - In Google Sheets: File → Download → CSV (UTF-8)
                    
                    **Required Columns:**
                    - `company` - Company name (required)
                    - `website` - Website URL (required) 
                    - `country` - Country location (required)
                    - `industry` - Industry sector (optional)
                    
                    **Data Format:**
                    ```csv
                    company,website,country,industry
                    Example Corp,example.com,Germany,Technology
                    Another Company,anothercompany.com,USA,Manufacturing
                    ```
                    
                    **Special Characters:**
                    - Avoid special characters in company names if possible
                    - Use UTF-8 encoding to support international characters
                    - Remove any extra spaces or hidden characters
                    """)
                
                # Show file info for debugging
                try:
                    uploaded_file.seek(0)
                    file_content = uploaded_file.read()
                    st.info(f"📁 File size: {len(file_content)} bytes")
                    
                    # Try to detect encoding
                    import chardet
                    detected = chardet.detect(file_content)
                    if detected:
                        st.info(f"🔍 Detected encoding: {detected.get('encoding', 'Unknown')} (confidence: {detected.get('confidence', 0):.2%})")
                except:
                    pass

def display_single_result(result, search_methods):
    """Display results for a single company"""
    if 'error' in result:
        st.error(f"Error processing {result.get('company', 'company')}: {result['error']}")
        return
    
    st.success(f"✅ Research completed for {result['company']}")
    
    # Web Scraping Results
    if "Website Scraping" in search_methods and result.get('web_scraping_data'):
        display_web_scraping_results(result['web_scraping_data'])
    
    # WHOIS Results
    if "WHOIS Lookup" in search_methods and result.get('whois_data'):
        with st.expander("📋 WHOIS Domain Information", expanded=True):
            whois_data = result['whois_data']
            
            # Check if there was an error
            if whois_data.get('error'):
                st.warning(f"⚠️ {whois_data['error']}")
                if whois_data.get('raw_data'):
                    st.text_area("Raw WHOIS Data", whois_data['raw_data'], height=100)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if whois_data.get('org'):
                        st.info(f"**Registered Organization**: {whois_data['org']}")
                    if whois_data.get('registrar'):
                        st.info(f"**Registrar**: {whois_data['registrar']}")
                    if whois_data.get('country'):
                        st.info(f"**Country**: {whois_data['country']}")
                
                with col2:
                    if whois_data.get('emails'):
                        st.info(f"**Contact Emails**: {', '.join(whois_data['emails'])}")
                    if whois_data.get('creation_date'):
                        st.info(f"**Domain Created**: {whois_data['creation_date']}")
                    
                # Show additional info if available
                if whois_data.get('name_servers'):
                    st.info(f"**Name Servers**: {', '.join(whois_data['name_servers']) if isinstance(whois_data['name_servers'], list) else whois_data['name_servers']}")
                if whois_data.get('expiration_date'):
                    st.info(f"**Domain Expires**: {whois_data['expiration_date']}")
    
    # AI Research Results
    if "AI Research" in search_methods and result.get('ai_research'):
        ai_provider_name = "Google Gemini" if result.get('ai_provider') == "gemini" else "OpenRouter"
        st.subheader(f"🧠 AI Research Results ({ai_provider_name})")
        
        with st.expander("📄 Full Research Report", expanded=True):
            st.markdown(result['ai_research'])
        
        # Parse and display structured data
        df = parse_ai_results_to_dataframe(result['ai_research'])
        if df is not None:
            st.subheader("📊 Structured Contact Data")
            st.dataframe(df, use_container_width=True)
            
            # Show contact count info
            contact_count = len(df)
            if contact_count > 1:
                st.success(f"🎯 Found {contact_count} contacts using {ai_provider_name}! In batch processing, this would create {contact_count} separate rows in your results CSV.")
            elif contact_count == 1:
                st.info(f"📧 Found 1 contact using {ai_provider_name}. Additional contacts may be found through website scraping.")
            
            # Export options
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_data = df.to_csv(index=False).encode("utf-8")
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_data = excel_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{result['company'].lower().replace(' ', '_')}_contacts_{timestamp}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"{result['company'].lower().replace(' ', '_')}_contacts_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

def process_batch_companies(provider, api_key, model, companies_df, search_methods, progress_callback=None):
    """Process multiple companies with progress tracking"""
    results = []
    total = len(companies_df)
    
    for idx, row in companies_df.iterrows():
        if progress_callback:
            progress_callback(idx + 1, total, f"Processing {row.get('company', 'Unknown')}")
        
        result = process_single_company(
            provider, api_key, model,
            row.get('company', ''),
            row.get('website', ''),
            row.get('country', ''),
            row.get('industry', ''),
            search_methods
        )
        results.append(result)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
    
    return results

def process_batch_csv(companies_df, provider, api_key, model, search_methods):
    """Process batch CSV with progress tracking"""
    st.subheader("🔄 Batch Processing Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, company_name):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {current}/{total}: {company_name}")
    
    # Process all companies
    results = process_batch_companies(
        provider, api_key, model, companies_df, search_methods, update_progress
    )
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.text("✅ Batch processing completed!")
    
    # Display results summary
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Companies Processed", len(results))
    with col2:
        st.metric("Successful", len(successful_results))
    with col3:
        st.metric("Failed", len(failed_results))
    
    # Show failed results
    if failed_results:
        with st.expander("❌ Failed Processes"):
            for result in failed_results:
                st.error(f"{result.get('company', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    # Combine all successful results into downloadable format
    if successful_results:
        all_contacts = []
        total_contacts_found = 0
        
        for result in successful_results:
            # Combine Web Scraping and AI results
            company_contacts = []
            
            # Add Web Scraping results
            if result.get('web_scraping_data'):
                scraping_data = result['web_scraping_data']
                for email in scraping_data.get('emails', []):
                    company_contacts.append({
                        'Name': '',
                        'Role': '',
                        'Email': email,
                        'Phone': '',
                        'Source': 'Website Scraping',
                        'Confidence': 'High',
                        'Company': result['company'],
                        'Website': result['website'],
                        'Country': result['country']
                    })
                for phone in scraping_data.get('phones', []):
                    company_contacts.append({
                        'Name': '',
                        'Role': '',
                        'Email': '',
                        'Phone': phone,
                        'Source': 'Website Scraping',
                        'Confidence': 'High',
                        'Company': result['company'],
                        'Website': result['website'],
                        'Country': result['country']
                    })
            
            # Add AI research results
            if result.get('ai_research'):
                df = parse_ai_results_to_dataframe(result['ai_research'])
                if df is not None:
                    for _, row in df.iterrows():
                        company_contacts.append({
                            'Name': row.get('Name', ''),
                            'Role': row.get('Role/Title', row.get('Role', '')),
                            'Email': row.get('Email', ''),
                            'Phone': row.get('Phone', ''),
                            'Source': row.get('Source', 'AI Research'),
                            'Confidence': row.get('Confidence', 'Medium'),
                            'Company': result['company'],
                            'Website': result['website'],
                            'Country': result['country']
                        })
            
            if company_contacts:
                all_contacts.extend(company_contacts)
                total_contacts_found += len(company_contacts)
        
        if all_contacts:
            combined_df = pd.DataFrame(all_contacts)
            
            st.subheader("📊 Combined Results")
            
            # Show summary of multiple contacts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Companies Input", len(successful_results))
            with col2:
                st.metric("Contact Rows Output", len(combined_df))
            with col3:
                avg_contacts = len(combined_df) / len(successful_results) if successful_results else 0
                st.metric("Avg Contacts/Company", f"{avg_contacts:.1f}")
            
            st.info("📋 **Multiple Contacts Structure**: Each contact gets its own row. Company information is repeated for each contact found.")
            
            st.dataframe(combined_df, use_container_width=True)
            
            # Export combined results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_data = combined_df.to_csv(index=False).encode("utf-8")
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='All Contacts', index=False)
                
                # Create summary sheet
                summary_df = pd.DataFrame({
                    'Company': [r['company'] for r in successful_results],
                    'Website': [r['website'] for r in successful_results],
                    'Country': [r['country'] for r in successful_results],
                    'Scraping Emails': [len(r.get('web_scraping_data', {}).get('emails', [])) for r in successful_results],
                    'Scraping Phones': [len(r.get('web_scraping_data', {}).get('phones', [])) for r in successful_results],
                    'AI Research': ['Yes' if r.get('ai_research') else 'No' for r in successful_results],
                    'Processed At': [r['processed_at'].strftime("%Y-%m-%d %H:%M") for r in successful_results]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            excel_data = excel_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download All Contacts (CSV)",
                    data=csv_data,
                    file_name=f"ai_batch_contacts_{timestamp}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download All Contacts (Excel)",
                    data=excel_data,
                    file_name=f"ai_batch_contacts_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Show breakdown of results
            with st.expander("📈 Detailed Results Breakdown"):
                
                # Count contacts by source
                source_counts = combined_df['Source'].value_counts()
                confidence_counts = combined_df['Confidence'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Contacts by Source")
                    for source, count in source_counts.items():
                        st.metric(source, count)
                
                with col2:
                    st.subheader("🎯 Contacts by Confidence")
                    for confidence, count in confidence_counts.items():
                        st.metric(f"{confidence} Confidence", count)
                
                # Show companies with most contacts
                st.subheader("🏆 Top Companies by Contacts Found")
                company_contact_counts = combined_df['Company'].value_counts().head(10)
                for company, count in company_contact_counts.items():
                    st.text(f"{company}: {count} contacts")
                
                # Show overall statistics
                st.subheader("📋 Summary Statistics")
                st.text(f"• Total companies processed: {len(successful_results)}")
                st.text(f"• Total contact rows generated: {len(combined_df)}")
                st.text(f"• Average contacts per company: {len(combined_df) / len(successful_results):.1f}")
                st.text(f"• Companies with 5+ contacts: {len(company_contact_counts[company_contact_counts >= 5])}")
                st.text(f"• Companies with 10+ contacts: {len(company_contact_counts[company_contact_counts >= 10])}")
            
            st.success(f"🎉 **Export Complete!** Your CSV contains {len(combined_df)} contact rows from {len(successful_results)} companies. Each contact is a separate row with full company details.")

    # Tips section
    with st.expander("💡 Tips & Best Practices"):
        st.markdown("""
        **🌐 Website Scraping:**
        - Extracts contact information directly from website pages
        - Respects robots.txt and follows ethical scraping guidelines
        - Searches contact pages, about pages, team directories
        - Extracts emails, phone numbers, names, and addresses
        
        **🧠 AI Research:**
        - Searches LinkedIn, business directories, news sources
        - Provides context and recent information
        - Best with Web Search models for real-time data
        - Cross-references multiple online sources
        
        **📋 WHOIS Lookup:**
        - Provides domain registration information
        - Technical and administrative contacts
        - Organization details and registration dates
        
        **✅ Streamlit Cloud Compatible:**
        - All features work reliably in Streamlit Cloud
        - No dependency conflicts or reactor issues
        - Scalable for batch processing
        - Secure API key handling through Streamlit secrets
        """)

if __name__ == "__main__":
    main()
