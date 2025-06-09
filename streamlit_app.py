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

try:
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    pass  # openpyxl styling is optional

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
            if e.response.status_code == 404:
                return None
            else:
                st.warning(f"HTTP error {e.response.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            if "timeout" in str(e).lower():
                st.warning(f"Timeout fetching {url}")
            return None
    
    def extract_contacts_from_text(self, text, url=""):
        """Extract contact information from text"""
        # Extract emails
        emails = self.email_pattern.findall(text)
        for email in emails:
            if not any(skip in email.lower() for skip in 
                      ['noreply', 'no-reply', 'donotreply', 'example.com', 'test.com']):
                self.contacts_found['emails'].add(email.lower())
        
        # Extract phone numbers
        phones = self.phone_pattern.findall(text)
        for phone in phones:
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) >= 7:
                self.contacts_found['phones'].add(phone.strip())
    
    def extract_names_from_soup(self, soup):
        """Extract potential contact names from BeautifulSoup object"""
        name_contexts = [
            'team', 'staff', 'contact', 'management', 'leadership',
            'ceo', 'cto', 'cfo', 'director', 'manager', 'head'
        ]
        
        for context in name_contexts:
            elements = soup.find_all(text=re.compile(context, re.I))
            for element in elements:
                parent = element.parent if element.parent else element
                text = parent.get_text() if hasattr(parent, 'get_text') else str(parent)
                names = self.extract_person_names(text)
                self.contacts_found['names'].update(names)
    
    def extract_person_names(self, text):
        """Extract person names using pattern matching"""
        names = set()
        name_pattern = r'\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(name_pattern, text)
        
        for match in matches:
            if not any(word.lower() in match.lower() for word in 
                      ['contact', 'email', 'phone', 'address', 'website', 'company']):
                if len(match.split()) >= 2:
                    names.add(match.strip())
        
        return names
    
    def extract_addresses_from_text(self, text):
        """Extract physical addresses"""
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[\s,]+[A-Za-z\s]+,?\s*\d{5}',
            r'[A-Za-z\s]+\d+\s*,\s*\d{5}\s+[A-Za-z\s]+',
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
        
        for link in links:
            href = link['href']
            link_text = (link.get_text() or '').lower()
            href_lower = href.lower()
            
            is_contact_link = (
                any(keyword in href_lower for keyword in self.contact_keywords) or
                any(keyword in link_text for keyword in ['contact', 'about', 'team', 'impressum'])
            )
            
            if is_contact_link:
                absolute_url = urljoin(base_url, href)
                contact_links.append(absolute_url)
        
        return list(set(contact_links))
    
    def scrape_website(self, start_url, max_pages=5):
        """Scrape website for contact information"""
        try:
            if not start_url.startswith(('http://', 'https://')):
                start_url = 'https://' + start_url
            
            domain = urlparse(start_url).netloc
            visited_pages = set()
            pages_found = 0
            
            progress_placeholder = st.empty()
            
            progress_placeholder.info(f"🏠 Analyzing homepage: {start_url}")
            
            homepage_response = self.fetch_page(start_url)
            if not homepage_response:
                st.warning(f"Could not access homepage: {start_url}")
                return None
            
            visited_pages.add(start_url)
            self.contacts_found['pages_scraped'].append(start_url)
            pages_found += 1
            
            homepage_soup = BeautifulSoup(homepage_response.content, 'html.parser')
            homepage_text = homepage_soup.get_text()
            
            self.extract_contacts_from_text(homepage_text, start_url)
            self.extract_names_from_soup(homepage_soup)
            self.extract_addresses_from_text(homepage_text)
            self.extract_social_links(homepage_soup)
            
            contact_links_found = self.find_contact_links(homepage_soup, start_url)
            
            progress_placeholder.info(f"🔍 Found {len(contact_links_found)} potential contact pages")
            
            for contact_url in contact_links_found[:max_pages-1]:
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
                
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                
                self.extract_contacts_from_text(text, contact_url)
                self.extract_names_from_soup(soup)
                self.extract_addresses_from_text(text)
                self.extract_social_links(soup)
                
                time.sleep(1)
            
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
                        continue
                    
                    visited_pages.add(fallback_url)
                    self.contacts_found['pages_scraped'].append(fallback_url)
                    self.contacts_found['contact_pages'].append(fallback_url)
                    pages_found += 1
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    
                    self.extract_contacts_from_text(text, fallback_url)
                    self.extract_names_from_soup(soup)
                    self.extract_addresses_from_text(text)
                    self.extract_social_links(soup)
                    
                    time.sleep(1)
            
            progress_placeholder.success(f"✅ Web scraping completed! Analyzed {pages_found} pages")
            
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
        
        return {
            "web_search": sorted(web_search_models),
            "free": sorted(free_models),
            "premium": sorted(premium_models)
        }
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return {"web_search": [], "free": [], "premium": []}


def create_comprehensive_search_prompt(company, website, country, industry=""):
    """Create comprehensive search prompt that searches ALL online sources"""
    domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    
    prompt = f"""Act as expert web researcher. Search on given website and every source like directory, social media publicly available online.

Please find the names and roles of the CEO OR Founder OR Managing Director OR Managing Partner for the company {company} with the website "{website}" based in {country}.

With the person you found, provide their personal LinkedIn URLs and work email addresses using the domain @{domain}. Find top one contact only. Also include the general company contact email.

Present the information in a table with name, role, full linkedin url, work email, general email, source with real link, confidence.

SEARCH ALL THESE SOURCES:
1. Company website ({website}) - Homepage, About page, Team page, Contact page, Leadership section
2. LinkedIn company page and executive profiles
3. Crunchbase company profile
4. ZoomInfo business directory
5. Apollo.io contact database
6. Hunter.io email finder
7. Google search results for "{company} CEO" and "{company} founder"
8. News articles and press releases mentioning company leadership
9. Business directories (Yellow Pages, Yelp Business, etc.)
10. Industry-specific directories and databases
11. Social media platforms (Twitter, Facebook business pages)
12. Company filings and regulatory documents
13. Interview articles and company announcements
14. Professional networking sites (XING for German companies)
15. Industry publications and trade magazines

REQUIREMENTS:
- Find REAL person names, not placeholders
- Provide REAL LinkedIn URLs that work (full https://linkedin.com/in/username format)
- Use actual email addresses following @{domain} pattern
- Include source with real, working link where information was found
- Rate confidence as High/Medium/Low based on source reliability
- If not found after comprehensive search, clearly state "NOT FOUND"
- Focus on current leadership (not former executives)

Example format:
| Name | Role | LinkedIn URL | Work Email | General Email | Source | Confidence |
|------|------|--------------|------------|---------------|--------|------------|
| John Smith | CEO | https://linkedin.com/in/johnsmith | j.smith@{domain} | info@{domain} | LinkedIn Profile | High |

Begin comprehensive multi-source research now."""

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
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
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
    
    gemini_available = check_gemini_availability()
    
    if openrouter_key:
        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", 
                               headers={"Authorization": f"Bearer {openrouter_key}"})
            if resp.status_code == 200:
                models = resp.json()["data"]
                
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
        except Exception as e:
            st.warning(f"Could not load OpenRouter models: {e}")
    
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
        st.warning("⚠️ Gemini API key provided but google-generativeai library not installed")
    
    return providers


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


def query_ai_enhanced(provider, api_key, model, prompt, timeout=120):
    """Enhanced AI query supporting multiple providers"""
    if provider == "openrouter":
        return query_openrouter_enhanced(api_key, model, prompt, timeout)
    elif provider == "gemini":
        return query_gemini_api(api_key, prompt, timeout)
    else:
        st.error(f"Unknown AI provider: {provider}")
        return None


def validate_and_clean_ai_response(ai_result, company_name):
    """Validate AI response and reject placeholder/lazy responses"""
    if not ai_result:
        return None, "No AI response received"
    
    placeholder_indicators = [
        "[Name", "[Email", "[Phone", "[Not Publicly", 
        "Name of Managing", "Name of Sales", "Name of Director",
        "Email from Impressum", "Phone from Impressum", "Phone from Website",
        "Not Publicly Available", "[URL]", "[Title]", "[Source]"
    ]
    
    placeholder_count = sum(1 for indicator in placeholder_indicators if indicator in ai_result)
    
    if placeholder_count > 2:
        return None, f"AI provided lazy placeholder response ({placeholder_count} placeholders found) - rejecting"
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    real_emails = re.findall(email_pattern, ai_result)
    
    real_emails = [email for email in real_emails if not any(placeholder in email for placeholder in ["example", "placeholder", "template"])]
    
    name_patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',
        r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',
    ]
    
    real_names = []
    for pattern in name_patterns:
        names = re.findall(pattern, ai_result)
        real_names.extend(names)
    
    generic_names = ["John Doe", "Jane Doe", "First Last", "Name Surname"]
    real_names = [name for name in real_names if name not in generic_names]
    
    if len(real_emails) == 0 and len(real_names) == 0:
        return None, "No real contact information found - only generic/placeholder content"
    
    quality_score = len(real_emails) + len(real_names)
    
    return ai_result, f"Response validated - found {len(real_emails)} real emails and {len(real_names)} real names"


def query_ai_with_validation(provider, api_key, model, prompt, company_name, timeout=120):
    """Query AI with validation and retry logic for better results"""
    max_attempts = 2
    
    for attempt in range(max_attempts):
        if attempt > 0:
            enhanced_prompt = f"""
SECOND ATTEMPT - PREVIOUS RESPONSE WAS REJECTED FOR PLACEHOLDER TEXT

{prompt}

CRITICAL: Your previous response contained placeholder text like [Name] or [Email]. This is UNACCEPTABLE.
Provide ONLY real, actual contact information or clearly state "NOT FOUND".
NO BRACKETS [] OR PLACEHOLDER TEXT ALLOWED.
"""
        else:
            enhanced_prompt = prompt
        
        ai_result = query_ai_enhanced(provider, api_key, model, enhanced_prompt, timeout)
        
        if not ai_result:
            continue
        
        validated_result, validation_message = validate_and_clean_ai_response(ai_result, company_name)
        
        if validated_result:
            st.success(f"✅ {validation_message}")
            return validated_result
        else:
            st.warning(f"⚠️ Attempt {attempt + 1}: {validation_message}")
            if attempt < max_attempts - 1:
                st.info("🔄 Retrying with stricter instructions...")
    
    st.error("❌ AI failed to provide real contact information after multiple attempts")
    return "AI_VALIDATION_FAILED: No real contact information could be extracted. The AI provided only placeholder text or generic responses."


def get_whois_contacts(domain):
    """Extract comprehensive contacts from WHOIS data"""
    try:
        w = None
        
        try:
            import whois
            if hasattr(whois, 'whois'):
                w = whois.whois(domain)
            elif hasattr(whois, 'query'):
                w = whois.query(domain)
        except:
            pass
        
        if w is None:
            try:
                result = subprocess.run(['whois', domain], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
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
            
            if hasattr(w, 'emails') and w.emails:
                if isinstance(w.emails, list):
                    contacts['emails'] = list(set(w.emails))
                else:
                    contacts['emails'] = [w.emails]
            
            for field in ['admin_email', 'tech_email', 'billing_email']:
                if hasattr(w, field):
                    email = getattr(w, field)
                    if email and email not in contacts['emails']:
                        contacts['emails'].append(email)
            
            return contacts
        
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
        
        if "WHOIS Lookup" in search_methods:
            results['whois_data'] = get_whois_contacts(domain)
        
        if "Website Scraping" in search_methods:
            scraper = WebScraper(company)
            results['web_scraping_data'] = scraper.scrape_website(website)
        
        if "AI Research" in search_methods:
            prompt = create_comprehensive_search_prompt(company, website, country, industry)
            results['ai_research'] = query_ai_with_validation(provider, api_
