import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Cấu hình môi trường và bảo mật
dotenv.config();
const app = express();
const port = process.env.PORT || 4000;

// Cấu hình giới hạn dữ liệu cho hình ảnh độ phân giải cao
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Khởi tạo Google AI với cấu hình an toàn
const API_KEY = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(API_KEY);

/**
 * MODULE: MEDICAL AI ENGINE
 * Cấu hình tham số cực thấp (Temperature = 0.1) để tránh AI sáng tạo sai kiến thức y khoa
 */
const medicalModel = genAI.getGenerativeModel({ 
    model: "gemini-2.5-flash",
    generationConfig: { 
        temperature: 0.1, 
        topP: 0.8, 
        topK: 40, 
        maxOutputTokens: 2048 
    }
});

// MIDDLEWARE: Quản lý phiên làm việc và Nhật ký y tế
const medicalTrafficLogger = (req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[MEDICAL-LOG] ${timestamp} | Request: ${req.method} | IP: ${req.ip}`);
    next();
};
app.use(medicalTrafficLogger);

/**
 * API ENDPOINT: Xử lý sàng lọc bệnh trạng chuyên sâu
 */
app.post("/api/v1/medical/analyze", async (req, res) => {
    try {
        const { prompt, image, history, patientProfile } = req.body;

        // PROMPT AI CHI TIẾT > 200 CHỮ (Linh hồn của hệ thống 100 triệu)
        const MASTER_PROMPT = `
        BẠN LÀ TRỢ LÝ Y TẾ AI CAO CẤP THUỘC HỆ THỐNG MEDMIND PRO (PHIÊN BẢN 2026).
        NHIỆM VỤ CỐ ĐỊNH: Phân tích triệu chứng lâm sàng qua văn bản và hình ảnh để thực hiện quy trình Triage (Phân loại ưu tiên cấp cứu).

        QUY TẮC PHẢN HỒI BẮT BUỘC:
        1. PHÂN LOẠI (LEVEL): Xác định mức độ dựa trên dấu hiệu sinh tồn ẩn ý.
           - [CRITICAL]: Đau thắt ngực, khó thở, liệt chi, mất máu ồ ạt, hôn mê.
           - [WARNING]: Sốt cao > 39 độ, đau bụng cấp, vết thương sâu, nhiễm trùng.
           - [STABLE]: Ho nhẹ, cảm mạo, dị ứng ngoài da nhẹ, đau cơ.
        2. CẤU TRÚC PHẢN HỒI (CHỈ TRẢ VỀ ĐỊNH DẠNG JSON):
           {
             "priority": "CRITICAL" | "WARNING" | "STABLE",
             "diagnosis_preview": "Chẩn đoán sơ bộ dựa trên dữ liệu",
             "clinical_reasoning": "Giải thích cơ chế bệnh sinh một cách khoa học ngắn gọn",
             "emergency_instructions": "Hành động khẩn cấp cần làm ngay (ví dụ: Ép tim, uống nước đường, gọi 115)",
             "next_investigation_question": "Câu hỏi chuyên sâu tiếp theo để loại trừ bệnh (duy nhất 1 câu)",
             "risk_score": "Số từ 1-100 thể hiện mức độ nguy hiểm",
             "vitals_prediction": {"heart_rate": "bpm", "temp": "độ C"}
           }
        3. GIỚI HẠN: Không được phép kê đơn thuốc đặc trị. Không lặp lại bất kỳ câu hỏi nào đã có trong lịch sử trò chuyện. 
        Nếu bệnh nhân gửi ảnh, hãy phân tích kỹ màu sắc, hình thái và liên kết trực tiếp với triệu chứng đã nêu.
        Mọi câu trả lời phải dựa trên phác đồ y khoa chuẩn quốc tế.
        `;

        let aiResult;
        if (image) {
            const imagePart = { 
                inlineData: { data: image.split(",")[1], mimeType: "image/jpeg" } 
            };
            aiResult = await medicalModel.generateContent([MASTER_PROMPT, prompt || "Phân tích dữ liệu hình ảnh", imagePart]);
        } else {
            const chatSession = medicalModel.startChat({ 
                history: history ? history.map(h => ({ role: h.role, parts: [{ text: h.text }] })) : [] 
            });
            aiResult = await chatSession.sendMessage(`${MASTER_PROMPT} | Input: ${prompt}`);
        }

        const responseText = (await aiResult.response).text();
        
        // Trích xuất JSON an toàn
        const jsonContent = responseText.match(/\{[\s\S]*\}/);
        if (!jsonContent) throw new Error("AI trả về định dạng không chuẩn");

        const parsedData = JSON.parse(jsonContent[0]);
        res.status(200).json({ ...parsedData, server_time: new Date() });

    } catch (error) {
        console.error("CRITICAL ERROR:", error.message);
        res.status(500).json({ 
            priority: "STABLE", 
            diagnosis_preview: "Hệ thống đang bảo trì logic phân tích.",
            emergency_instructions: "Vui lòng liên hệ cơ sở y tế gần nhất nếu tình trạng xấu đi."
        });
    }
});

app.listen(port, () => {
  console.log(`Server đang chạy ở port ${port}`);
});