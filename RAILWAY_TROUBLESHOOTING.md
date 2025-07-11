# 🚂 Railway Deployment Troubleshooting Guide

## 🔧 COMMON RAILWAY ISSUES & FIXES

### **Issue 1: Build Failures**

**Symptom**: Build fails with dependency errors
**Fix**: Create minimal requirements for Railway

```bash
# Create Railway-specific requirements
cat > requirements.railway.txt << 'EOF'
requests>=2.31.0
aiohttp>=3.8.0
flask>=2.3.0
psutil>=5.9.0
websocket-client>=1.6.0
EOF

# Commit and push
git add requirements.railway.txt
git commit -m "Add Railway requirements"
git push
```

### **Issue 2: Port Binding Problems**

**Symptom**: "Port already in use" or "Cannot bind to port"
**Fix**: Ensure proper PORT environment variable handling

```python
# Verify mock_qnti_server.py has this code:
import os
port = int(os.environ.get('PORT', 5000))
host = '0.0.0.0'  # Important for Railway!
```

### **Issue 3: Start Command Issues**

**Railway Settings Fix**:
1. Go to your Railway service settings
2. **Variables** tab → Add:
   - `PORT`: (leave empty, Railway auto-assigns)
   - `PYTHON_VERSION`: `3.11`
3. **Settings** tab → **Deploy**:
   - **Start Command**: `python mock_qnti_server.py`
   - **Build Command**: `pip install -r requirements.txt`

### **Issue 4: Memory/Resource Limits**

**Symptom**: "Out of memory" or crashes
**Fix**: Use minimal imports in production

```python
# Add to top of mock_qnti_server.py
import os
if os.environ.get('RAILWAY_ENVIRONMENT'):
    # Minimal imports for Railway
    pass
```

### **Issue 5: File Not Found Errors**

**Symptom**: "ModuleNotFoundError" or missing files
**Fix**: Check file structure and imports

```bash
# Verify your files are committed
git ls-files | grep -E "(mock_qnti_server|requirements)"
```

## 🚀 QUICK FIX SOLUTIONS

### **Solution A: Railway-Optimized Server**

Create a Railway-specific server file:

```python
# railway_server.py
import asyncio
import os
from aiohttp import web
import json
from datetime import datetime

async def health_check(request):
    return web.json_response({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'QNTI System Running on Railway'
    })

async def dashboard(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>QNTI Trading System</title></head>
    <body style="font-family: Arial; background: #1a1a1a; color: #fff; padding: 20px;">
        <h1>🚀 QNTI Trading System</h1>
        <h2>✅ Successfully Deployed on Railway!</h2>
        <div style="background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px;">
            <h3>System Status</h3>
            <p>✅ Server Running</p>
            <p>✅ API Endpoints Active</p>
            <p>✅ Railway Deployment Successful</p>
        </div>
        <div style="background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px;">
            <h3>Available Endpoints</h3>
            <p><a href="/api/health" style="color: #4CAF50;">/api/health</a> - Health Check</p>
            <p><a href="/api/ea/indicators" style="color: #4CAF50;">/api/ea/indicators</a> - EA Indicators</p>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def ea_indicators(request):
    return web.json_response({
        'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands'],
        'count': 5,
        'status': 'Railway deployment successful'
    })

def create_app():
    app = web.Application()
    app.router.add_get('/', dashboard)
    app.router.add_get('/api/health', health_check)
    app.router.add_get('/api/ea/indicators', ea_indicators)
    return app

async def main():
    app = create_app()
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🚀 Starting QNTI on Railway - Port {port}")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    print(f"✅ QNTI System running on Railway!")
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
```

### **Solution B: Minimal Requirements**

```bash
# Create minimal requirements
echo "aiohttp>=3.8.0" > requirements.txt
git add requirements.txt
git commit -m "Minimal Railway requirements"
git push
```

## 📋 RAILWAY CONFIGURATION CHECKLIST

### **In Railway Dashboard:**

1. **Service Settings**:
   - ✅ **Source**: GitHub repository connected
   - ✅ **Branch**: main
   - ✅ **Root Directory**: / (default)

2. **Build Settings**:
   - ✅ **Build Command**: `pip install -r requirements.txt`
   - ✅ **Start Command**: `python mock_qnti_server.py`

3. **Environment Variables**:
   - ✅ **PORT**: (auto-assigned by Railway)
   - ✅ **PYTHON_VERSION**: 3.11

4. **Networking**:
   - ✅ **Public Domain**: Enabled
   - ✅ **Custom Domain**: (optional)

## 🆘 EMERGENCY DEPLOYMENT ALTERNATIVES

### **If Railway Keeps Failing - Use Render:**

1. **Go to**: https://render.com/
2. **New** → **Web Service**
3. **Connect** your GitHub repo
4. **Settings**:
   - **Name**: qnti-trading-system
   - **Build Command**: `pip install aiohttp`
   - **Start Command**: `python mock_qnti_server.py`
5. **Deploy**

### **Or Try Vercel (Serverless):**

1. **Go to**: https://vercel.com/
2. **Import Project** from GitHub
3. **Framework**: Other
4. **Deploy** (auto-configured)

## 🔍 DEBUGGING STEPS

### **Check Railway Logs**:
1. Railway Dashboard → **Deployments**
2. Click latest deployment
3. **View Logs** to see exact error
4. Look for:
   - Import errors
   - Port binding issues
   - Memory problems
   - File not found errors

### **Common Log Errors & Fixes**:

**Error**: `ModuleNotFoundError: No module named 'X'`
**Fix**: Add module to requirements.txt

**Error**: `Port already in use`
**Fix**: Ensure using `os.environ.get('PORT')`

**Error**: `Permission denied`
**Fix**: Ensure using `0.0.0.0` as host

## 🚀 QUICK TEST

After fixing, test your deployment:

```bash
# Test health endpoint
curl https://your-railway-app.railway.app/api/health

# Should return JSON with status: healthy
```

---

**📋 Tell me what specific error you're seeing in Railway logs and I'll give you the exact fix!**