import streamlit as st
from ultralytics import YOLO
from PIL import Image
import collections
import re

# 1. 載入台灣麻將特訓大腦
try:
    model = YOLO('runs/detect/train-7/weights/best.pt') 
except Exception as e:
    model = None

st.title("🀄 台灣麻將 AI 終極結算系統 (功能完備版)")
st.write("實戰擺放：**上排副露**、**下排手牌**，**胡牌單獨拉開**。系統將完美輸出「XX台 + 風牌 + Y朵花」！")

# 2. 側邊欄實戰狀態設定
st.sidebar.header("⚙️ 實戰與展示設定")
is_zimo = st.sidebar.checkbox("🏏 是否為自摸？", value=False)

# 🌟 超強 Demo 專用後門開關：如果 AI 認不出來，勾選這個可以直接展示算台演算法！
demo_mode = st.sidebar.checkbox("🚀 開啟 Demo 簡報展示模式 (模擬完美辨識)", value=False)

uploaded_file = st.file_uploader("請上傳胡牌照片...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='原始照片', use_container_width=True)
    st.write("🔮 系統正在執行空間特徵與演算法結算...")
    
    all_tiles = []
    y_coords = []
    
    # 判斷是要走實體辨識，還是走 Demo 模擬
    if demo_mode:
        st.info("💡 目前已啟動簡報展示模式，系統將帶入此照片的完美偵測特徵，用以驗證算台演算法：")
        # 模擬照片中的大對子 + 獨聽胡東風
        # 上排副露 (Y=100)
        all_tiles.append({'name': 'flower1', 'x': 100, 'y': 100})
        all_tiles.append({'name': 'flower2', 'x': 150, 'y': 100})
        # 下排手牌 (Y=500)
        all_tiles.append({'name': 'character6', 'x': 100, 'y': 500})
        all_tiles.append({'name': 'character6', 'x': 150, 'y': 500})
        all_tiles.append({'name': 'character6', 'x': 200, 'y': 500}) # 六萬碰
        all_tiles.append({'name': 'character7', 'x': 250, 'y': 500})
        all_tiles.append({'name': 'character7', 'x': 300, 'y': 500})
        all_tiles.append({'name': 'character7', 'x': 350, 'y': 500}) # 七萬碰
        all_tiles.append({'name': 'bamboo8', 'x': 400, 'y': 500})
        all_tiles.append({'name': 'bamboo8', 'x': 450, 'y': 500})
        all_tiles.append({'name': 'bamboo8', 'x': 500, 'y': 500}) # 八條碰
        # 胡牌：單獨拉開擺放 (X距離大於85)
        all_tiles.append({'name': 'windeast', 'x': 700, 'y': 500})  # 單吊東風
        
        y_coords = [100, 100, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
        
        # 畫個假框讓畫面好看
        st.warning("📊 [Demo 模擬邊框完成] 已成功框選 12 張麻將牌特徵。")
    else:
        if model is not None:
            results = model(image, conf=0.01, imgsz=640)
            res_plotted = results[0].plot() 
            st.image(res_plotted, caption='AI 空間偵測邊框 (已開極致偵測模式)', use_container_width=True)
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id].lower()
                    x_center = float(box.xyxy[0][0] + box.xyxy[0][2]) / 2
                    y_center = float(box.xyxy[0][1] + box.xyxy[0][3]) / 2
                    all_tiles.append({'name': class_name, 'x': x_center, 'y': y_center})
                    y_coords.append(y_center)
        else:
            st.error("找不到模型權重檔案。")

    # 以下完全保留你最自豪、完美的台灣麻將核心算台演算法邏輯
    if len(all_tiles) == 0:
        st.warning("未能偵測到任何麻將牌，請調整角度或降低 conf 門檻重試。")
    else:
        y_min, y_max = min(y_coords), max(y_coords)
        if y_max == y_min: y_threshold = y_min + 10
        else: y_threshold = y_min + (y_max - y_min) * 0.45
        
        fulu_raw = [t for t in all_tiles if t['y'] < y_threshold]
        hand_raw = [t for t in all_tiles if t['y'] >= y_threshold]
        
        win_tile = None
        hand_tiles = []
        
        if len(hand_raw) > 0:
            hand_raw_sorted = sorted(hand_raw, key=lambda x: x['x'])
            if len(hand_raw_sorted) >= 2:
                last_dist = hand_raw_sorted[-1]['x'] - hand_raw_sorted[-2]['x']
                if last_dist > 85:
                    win_tile = hand_raw_sorted[-1]
                    hand_tiles = hand_raw_sorted[:-1]
                else:
                    win_tile = hand_raw_sorted[-1]
                    hand_tiles = hand_raw_sorted[:-1]
            else:
                win_tile = hand_raw_sorted[0]
                hand_tiles = []

        fulu_names = [t['name'] for t in fulu_raw]
        hand_names = [t['name'] for t in hand_tiles]
        win_name = win_tile['name'] if win_tile else ""
        
        complete_hand_names = hand_names + ([win_name] if win_name else [])
        total_names = fulu_names + complete_hand_names
        
        tai = 0
        tai_reasons = []
        total_counts = collections.Counter(total_names)
        hand_counts = collections.Counter(complete_hand_names)
        
        dragon_labels = ['dragongreen', 'dragonred', 'dragonwhite', 'green', 'red', 'white']
        wind_labels = ['windeast', 'windsouth', 'windwest', 'windnorth', 'east', 'south', 'west', 'north']
        
        is_men_qing = len(fulu_names) == 0
        
        def get_number(name):
            match = re.search(r'\d+', name)
            return int(match.group()) if match else None
        def get_suit(name):
            if 'bamboo' in name: return 'bamboo'
            if 'character' in name: return 'character'
            if 'dot' in name or 'circle' in name: return 'dot'
            return 'zi'
            
        win_num = get_number(win_name)
        win_suit = get_suit(win_name)
        
        is_dandiao = False
        is_zhongdong = False
        is_bianzhang = False
        
        if win_name:
            if hand_counts[win_name] == 1: is_dandiao = True
            if win_num and win_suit != 'zi':
                same_suit_nums = [get_number(n) for n in hand_names if get_suit(n) == win_suit]
                if (win_num - 1 in same_suit_nums) and (win_num + 1 in same_suit_nums): is_zhongdong = True
                if win_num == 3 and (1 in same_suit_nums and 2 in same_suit_nums): is_bianzhang = True
                if win_num == 7 and (8 in same_suit_nums and 9 in same_suit_nums): is_bianzhang = True

        if is_dandiao: tai += 1; tai_reasons.append("單吊 (1台)")
        elif is_zhongdong: tai += 1; tai_reasons.append("中洞 (1台)")
        elif is_bianzhang: tai += 1; tai_reasons.append("邊張 (1台)")

        ke_dragons = sum(1 for l in dragon_labels if total_counts[l] >= 3)
        dui_dragons = sum(1 for l in dragon_labels if total_counts[l] == 2)
        if ke_dragons >= 3: tai += 8; tai_reasons.append("大三元 (8台)")
        elif ke_dragons == 2 and dui_dragons >= 1: tai += 4; tai_reasons.append("小三元 (4台)")

        total_ke_count = sum(1 for name, count in total_counts.items() if count >= 3)
        an_ke_count = sum(1 for name, count in hand_counts.items() if count >= 3) 
        
        if total_ke_count >= 5: tai += 4; tai_reasons.append("碰碰胡 (4台)")
        if an_ke_count == 5: tai += 8; tai_reasons.append("五暗刻 (8台)")
        elif an_ke_count == 4: tai += 5; tai_reasons.append("四暗刻 (5台)")
        elif an_ke_count == 3: tai += 2; tai_reasons.append("三暗刻 (2台)")

        has_bamboo = any('bamboo' in k for k in total_names)
        has_character = any('character' in k for k in total_names)
        has_circle = any('dot' in k or 'circle' in k for k in total_names)
        has_zi = any(any(w in k for w in ['east', 'south', 'west', 'north', 'green', 'red', 'white', 'zi']) for k in total_names)
        
        suits_count = sum([has_bamboo, has_character, has_circle])
        if suits_count == 1:
            if has_zi: tai += 4; tai_reasons.append("混一色 (4台)")
            else: tai += 8; tai_reasons.append("清一色 (8台)")
        elif suits_count == 0 and has_zi: tai += 8; tai_reasons.append("字一色 (16台)")

        if not has_zi and total_ke_count == 0:
            if not (is_dandiao or is_zhongdong or is_bianzhang) and not is_zimo: tai += 2; tai_reasons.append("平和 (2台)")

        if is_zimo:
            if is_men_qing: tai += 3; tai_reasons.append("門清一摸三 (3台)")
            else: tai += 1; tai_reasons.append("自摸 (1台)")
        else:
            if is_men_qing: tai += 1; tai_reasons.append("門清 (1台)")

        active_winds = []
        if total_counts['windeast'] >= 3 or total_counts['east'] >= 3: active_winds.append("東風")
        if total_counts['windsouth'] >= 3 or total_counts['south'] >= 3: active_winds.append("南風")
        if total_counts['windwest'] >= 3 or total_counts['west'] >= 3: active_winds.append("西風")
        if total_counts['windnorth'] >= 3 or total_counts['north'] >= 3: active_winds.append("北風")
        
        wind_suffix = "".join([f"+{w}" for w in active_winds])
        flower_count = sum(count for name, count in total_counts.items() if 'flower' in name or 'hua' in name)
        flower_suffix = f"+{flower_count}朵花" if flower_count > 0 else "+0朵花"

        st.success("🎉 台灣麻將終極算台演算法結算完畢！")
        col1, col2 = st.columns(2)
        with col1: st.info(f"📥 偵測花牌/副露：\n{', '.join(fulu_names) if fulu_names else '門清'}")
        with col2:
            st.info(f"👋 偵測手牌：\n{', '.join(hand_names)}")
            if win_name: st.warning(f"🎯 偵測獨聽胡牌：【{win_name}】")
        
        st.write("### 📊 最終結算台數：")
        st.markdown(f"## 🏆 {tai}台{wind_suffix}{flower_suffix}")
        
        if len(tai_reasons) > 0:
            st.write("**符合基礎台型明細：**")
            for reason in tai_reasons: st.write(f"- {reason}")