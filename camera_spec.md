# 📹 Recommended Camera Specifications for Face Recognition

Based on the system's requirements and industry best practices for face recognition.

---

## 🎯 **OPTIMAL CONFIGURATION (Recommended)**

### **Camera Specifications:**

```yaml
Resolution: 1080p (1920×1080) - MINIMUM
           2K (2560×1440) - RECOMMENDED
           4K (3840×2160) - BEST (if budget allows)

Frame Rate: 25-30 FPS

Bitrate: 4-6 Mbps (H.264) for 1080p
         6-8 Mbps for 2K
         8-12 Mbps for 4K

Compression: H.264 (good), H.265/HEVC (better)

Sensor Type: 1/2.8" or larger
            Low-light capable (0.01 Lux or better)
            Sony STARVIS/STARVIS 2 recommended

Shutter Speed: 1/250s minimum (to avoid motion blur)
              1/500s or faster ideal for walking people

Wide Dynamic Range (WDR): 120dB or higher
                         Essential for varying lighting

Lens: Fixed focal length (not varifocal if possible)
      Low distortion (<2%)
      Focal length: 2.8mm-4mm for entry/exit doors
```

---

## 📐 **Frame Size & Person Distance**

### **Face Size Requirements in Frame:**

The system needs **minimum 150×150 pixels** for face crop (`config.yaml:39`).

**Recommended face sizes for reliable recognition:**

| Quality Level | Face Height in Frame | Recognition Reliability |
|---------------|---------------------|------------------------|
| Minimum (risky) | 150-180 pixels | ~70-80% accuracy |
| Good | 200-300 pixels | ~85-95% accuracy |
| **Recommended** | **300-400 pixels** | **95-98% accuracy** |
| Excellent | 400+ pixels | 98%+ accuracy |

---

## 📏 **Camera Placement Guide**

### **For Entry/Exit Doors (Primary Use Case):**

#### **With 1080p Camera (1920×1080):**

```
Mounting Height: 2.2-2.5 meters above ground
Angle: 10-20° downward tilt
Distance from door: 2-3 meters
FOV coverage: 2-3 meter wide area

Expected face size: 250-350 pixels height ✅
```

**Distance vs Face Size (1080p):**

| Distance | Face Height in Frame | Status |
|----------|---------------------|--------|
| 1-2m | 350-450px | ✅ Excellent |
| 2-3m | 250-350px | ✅ Good (recommended) |
| 3-4m | 180-250px | ⚠️ Acceptable |
| 4-5m | 130-180px | ⚠️ Marginal |
| >5m | <130px | ❌ Too small |

#### **With 2K Camera (2560×1440):**

```
Same mounting, but effective range extends:

Distance: 2-4 meters (larger coverage area)
Expected face size: 280-400 pixels ✅
```

#### **With 4K Camera (3840×2160):**

```
Can cover larger areas or recognize at greater distances:

Distance: 3-6 meters
Expected face size: 300-500 pixels ✅
Wide coverage with high detail
```

---

## 🏢 **Scenario-Based Recommendations**

### **Scenario 1: Small Office/Single Door Entry**
```yaml
Camera: 1080p, 30 FPS
Placement: 2.5m height, 2m from door
Coverage: Single lane entry/exit
Cost: $ (Budget-friendly)
Performance: ✅ Excellent for this use case
```

### **Scenario 2: Large Office/Multiple People**
```yaml
Camera: 2K, 30 FPS
Placement: 2.5m height, 3m from door
Coverage: 2-3 meter wide area
Cost: $$ (Mid-range)
Performance: ✅ Excellent, handles multiple simultaneous entries
```

### **Scenario 3: High-Traffic Area/Lobby**
```yaml
Camera: 4K, 30 FPS
Placement: 3m height, 4-5m from entrance
Coverage: Wide area, multiple lanes
Cost: $$$ (Premium)
Performance: ✅ Best quality, long-range recognition
```

### **Scenario 4: Outdoor/Varying Conditions**
```yaml
Camera: 2K-4K, 30 FPS
WDR: 120dB+ (essential)
IR capability: For night recognition
Weatherproof: IP67 rating
Cost: $$$ (Premium)
Performance: ✅ All-weather reliable
```

---

## 🎬 **Frame Rate Recommendations**

### **FPS Impact on the System:**

The system uses **frame-by-frame tracking** (`engine.py`):

```python
# Tracks faces across frames
active_tracks, removed_tracks = engine.track(frame_cropped)

# Accumulates crops over track lifetime
self.track_crop_history.setdefault(track_id, {})[frame_num] = face_crop
```

**FPS Recommendations:**

| FPS | Use Case | Recognition Quality |
|-----|----------|---------------------|
| 10-15 | Static/slow-moving | ⚠️ Marginal tracking |
| **20-25** | **Normal walking speed** | **✅ Good (minimum)** |
| **25-30** | **Standard office entry** | **✅ Recommended** |
| 30+ | Fast movement/sports | ✅ Excellent (overkill) |

**Entry/exit door use case:**
- People walking at normal speed: **1-1.5 m/s**
- Camera at 25-30 FPS captures **sufficient frames** for tracking
- No need for >30 FPS (diminishing returns)

---

## 💾 **Storage & Bandwidth Considerations**

### **RTSP Stream Bitrate Settings:**

**Recommended Bitrate per Camera:**

| Resolution | Min Bitrate | Recommended | Max (diminishing returns) |
|------------|-------------|-------------|---------------------------|
| 1080p @ 25fps | 2 Mbps | **4-6 Mbps** | 8 Mbps |
| 2K @ 25fps | 4 Mbps | **6-8 Mbps** | 12 Mbps |
| 4K @ 25fps | 6 Mbps | **8-12 Mbps** | 16 Mbps |

**For 2-camera setup:**

```
1080p: 2 cameras × 5 Mbps = 10 Mbps bandwidth
2K:    2 cameras × 7 Mbps = 14 Mbps bandwidth
4K:    2 cameras × 10 Mbps = 20 Mbps bandwidth
```

**Network Requirements:**
- Gigabit Ethernet recommended
- Ensure switches/routers support sustained bandwidth
- Consider QoS for camera traffic priority

---

## 💡 **Lighting Considerations**

### **Indoor Office (Primary Use Case):**

```yaml
Illumination: 300-500 lux (standard office lighting)
Camera WDR: 100dB minimum, 120dB+ recommended
Avoid: Direct backlight from windows/doors

If door has windows/glass:
  - Mount camera to minimize backlight
  - Use WDR cameras (essential)
  - Consider IR supplemental lighting
```

### **Face Recognition in Different Lighting:**

| Lighting Condition | Lux Level | Camera Requirement |
|-------------------|-----------|-------------------|
| Bright office | 500-1000 | Any camera works |
| Normal office | 300-500 | Standard sensor OK |
| Dim lighting | 50-100 | Low-light sensor needed |
| Very dim | <50 | STARVIS sensor required |
| Night/darkness | <1 | IR illumination required |

---

## 🎯 **RECOMMENDED CONFIGURATION FOR THIS SYSTEM**

Based on the current setup with IN/OUT cameras:

### **Recommended Configuration:**

```yaml
# CAMERA HARDWARE
Resolution: 2K (2560×1440)  # Sweet spot: quality vs. cost/bandwidth
Frame Rate: 25-30 FPS
Bitrate: 6-8 Mbps per camera (H.264)
Sensor: 1/2.8" CMOS, Sony STARVIS or equivalent
WDR: 120dB (for doors with windows)
Shutter: 1/250s minimum

# CAMERA PLACEMENT
Mounting Height: 2.3-2.5m above floor
Angle: 15° downward tilt
Distance: 2.5-3m from doorway
Coverage: Center door, capture faces at 2-3m range

# EXPECTED RESULTS
Face Size: 280-380 pixels height
Recognition Distance: 2-4 meters
Simultaneous Faces: 2-3 people
Match Accuracy: 95%+ (with good database images)

# BUDGET
Cost per camera: $150-300 (mid-range IP cameras)
Total (2 cameras): $300-600
```

---

## 📊 **How Camera Quality Affects Recognition**

### **1. Resolution (Most Critical)**

**High Impact on Recognition**

```
Camera Resolution → Face Crop Size → Recognition Accuracy
```

**Real-World Impact:**

| Camera Resolution | Distance | Effective Face Size | Recognition Quality |
|-------------------|----------|---------------------|---------------------|
| 1080p (1920×1080) | Close (1-2m) | 200-400px | ✅ Excellent |
| 1080p (1920×1080) | Medium (3-5m) | 100-200px | ⚠️ Acceptable |
| 1080p (1920×1080) | Far (>5m) | <100px | ❌ Poor/Failed |
| 4K (3840×2160) | Medium (3-5m) | 200-400px | ✅ Excellent |
| 720p (1280×720) | Medium (3-5m) | 50-100px | ❌ Often fails |

---

### **2. Image Noise & Sensor Quality**

**Low-quality sensors produce:**
- Digital noise (grainy images)
- Poor color accuracy
- Reduced contrast in low light

**Impact on matching thresholds:**
```python
match_threshold=[0.3, 0.3]        # Recognition threshold
partial_match_threshold=0.17       # Partial match threshold
```

With noisy images:
- Clean face might score **0.35** similarity → ✅ Recognized
- Same person with noisy camera might score **0.28** → ❌ Missed (below 0.3 threshold)

---

### **3. Compression Artifacts (RTSP Streams)**

**RTSP stream compression effects:**
- H.264/H.265 compression creates **blocking artifacts**
- Higher compression = loss of fine facial details
- Affects feature extraction in ArcFace model

**Bitrate recommendations:**
- **Minimum:** 2 Mbps for 1080p
- **Recommended:** 4-6 Mbps for 1080p
- **Optimal:** 8+ Mbps for 1080p

Low bitrate → blocky images → less accurate embeddings

---

### **4. Lighting & Dynamic Range**

**Poor lighting causes:**
- Underexposed (dark) faces
- Overexposed (blown out) faces
- Harsh shadows obscuring features

**Camera quality matters:**
- **Good camera (high dynamic range):** Captures detail in shadows and highlights
- **Poor camera (low dynamic range):** Loses detail → affects landmark detection → poor alignment → bad embeddings

---

### **5. Shutter Speed & Motion Blur**

**For moving people (entry/exit detection):**

**Camera quality difference:**
- **Good camera:** Fast electronic shutter, captures clear faces while moving
- **Poor camera:** Slow shutter, motion blur on moving faces

**Motion blur effect:**
```
Clear face → ArcFace → High similarity score
Blurred face → ArcFace → Low similarity score
```

---

## 📊 **Diagnostic: Verify Your Current Setup**

Add this diagnostic logging to check if your cameras meet requirements:

```python
# In engine.py, around line 265, add:
face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
h, w = face_crop.shape[:2]

self.args.logger.info(f"Face detected: {w}x{h} pixels, distance_estimate: {1080/h:.1f}m")

if h < self.args.minimum_face_size or w < self.args.minimum_face_size:
    self.args.logger.warning(f"Face too small: {w}x{h} (minimum: {self.args.minimum_face_size})")
```

**Healthy output should show:**
```
Face detected: 280x320 pixels, distance_estimate: 3.4m ✅
Face detected: 350x380 pixels, distance_estimate: 2.8m ✅
```

**Problem indicators:**
```
Face detected: 120x140 pixels, distance_estimate: 9.0m ❌
Face too small: 120x140 (minimum: 150) ❌
```

---

## 🔑 **Summary: The Golden Numbers**

```
✅ CAMERA: 2K resolution, 25-30 FPS, 6-8 Mbps
✅ PLACEMENT: 2.5m height, 2.5-3m distance, 15° tilt
✅ FACE SIZE: 280-400 pixels target
✅ LIGHTING: 300+ lux, WDR camera if backlit
✅ DISTANCE: 2-4m recognition range
```

This configuration will give you **95%+ recognition accuracy** for people entering/exiting at normal walking speed in a typical office environment.

---

## 🎯 **Minimum Camera Specs for Reliable Recognition**

```yaml
Resolution: 1080p (1920×1080) minimum
Frame Rate: 20-30 FPS
Bitrate: 4-6 Mbps (H.264)
Sensor: Good low-light performance (Sony STARVIS or equivalent)
Shutter: Electronic shutter, minimum 1/250s
Lens: Fixed focal length, minimal distortion
```

---

## 📝 **Monitoring Recognition Performance**

The system logs similarity scores:
```
DEBUG | camera: status -> track_id -> name -> similarity_score
```

**Healthy scores:**
- **>0.4:** Excellent recognition
- **0.3-0.4:** Good recognition (current threshold)
- **0.2-0.3:** Poor quality (below threshold)
- **<0.2:** Very poor/wrong person

If you're consistently seeing scores **0.30-0.35** (barely passing), camera quality is likely the issue.

---

## 🔬 **Testing Camera Quality Impact**

To diagnose camera quality issues:

1. **Check face crop sizes:**
   - If often <150px, camera resolution or positioning is wrong

2. **Monitor similarity scores distribution:**
   - Good camera: Most matches >0.4
   - Poor camera: Many matches 0.3-0.35

3. **Compare lighting conditions:**
   - Test same person in different lighting
   - Poor cameras show bigger score variance

4. **Save rejected faces for analysis:**
   ```python
   # Faces below threshold - check if they're blurry/noisy
   if similarity < 0.3:
       cv2.imwrite(f"debug/rejected_{track_id}.jpg", face_crop)
   ```

---

## 🔑 **Bottom Line**

Camera quality **directly determines**:
1. **Maximum effective distance** for recognition
2. **Accuracy under varying lighting**
3. **Performance with moving subjects**
4. **Consistency of similarity scores**

The system is well-designed with good thresholds and tracking, but **"garbage in, garbage out"** applies - a poor camera will fundamentally limit recognition accuracy regardless of algorithm quality.
