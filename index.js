import express from "express";
import multer from "multer";
import cors from "cors";
import natural from "natural";
import { default as pdfParse } from "pdf-parse/lib/pdf-parse.js";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatMistralAI } from "@langchain/mistralai";
import dotenv from "dotenv";
import { z } from "zod";
import { StructuredOutputParser } from "@langchain/core/output_parsers";

dotenv.config();
const app = express();
const PORT = 3000;

// Initialize Mistral AI
const mistralModel = new ChatMistralAI({
  apiKey: process.env.MISTRAL_API_KEY,
  modelName: "mistral-medium",
  temperature: 0.7,
  maxTokens: 2048,
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024,
    files: 1,
  },
});

app.use(cors());
app.use(express.json());

// Interview session store (in-memory, replace with DB in production)
const sessions = new Map();

// Enhanced keyword extraction
function extractKeywords(text, topN = 15) {
  const tokenizer = new natural.WordTokenizer();
  const words = tokenizer.tokenize(text.toLowerCase());

  const keywords = words
    .filter(
      (word) =>
        word.length > 3 &&
        !natural.stopwords.includes(word) &&
        !/\d/.test(word) &&
        !["http", "https", "com"].includes(word)
    )
    .reduce((acc, word) => {
      const stemmed = natural.PorterStemmer.stem(word);
      acc[stemmed] = (acc[stemmed] || 0) + 1;
      return acc;
    }, {});

  return Object.entries(keywords)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([word]) => word);
}

// Interview prompt templates
const initialQuestionsPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an expert technical interviewer for {position} positions.
Skills: {keywords}

Generate 5-6 technical interview questions that:
1. Cover both fundamentals and advanced topics
2. Progress from easy to hard
3. Include at least 1 system design question
4. Relate to the mentioned skills

Format as a numbered list. Do NOT include any markdown.`,
  ],
  ["human", "Generate the questions now."],
]);

const interviewPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Generate {numQuestions} technical interview questions for a {position} role.
Focus on these skills: {keywords}

Format as a numbered list. Include a mix of:
- Conceptual questions
- Practical problems
- System design challenges`,
  ],
  ["human", "Generate the questions now."],
]);

// Define output schema for evaluation
const evaluationParser = StructuredOutputParser.fromZodSchema(
  z.object({
    score: z.number().min(1).max(4),
    rating: z.enum(["Poor", "Fair", "Good", "Excellent"]),
    feedback: z.string(),
    followUp: z.string()
  })
);

const evaluationPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Evaluate this technical interview answer:
Question: {question}
Answer: {answer}

Respond with JSON containing:
- score (1-4)
- rating (Poor/Fair/Good/Excellent)
- feedback (brief justification)
- followUp (relevant question)

{format_instructions}`,
  ],
  ["human", "Evaluate this answer strictly following the format."]
]);

// API Endpoints

// PDF Processing
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const data = await pdfParse(req.file.buffer);
    const keywords = extractKeywords(data.text);

    res.json({
      success: true,
      keywords,
      textSample: data.text.substring(0, 200) + "...",
    });
  } catch (error) {
    console.error("PDF Processing Error:", error);
    res.status(500).json({ error: "Failed to process document" });
  }
});

// Start Interview
app.post("/start-interview", express.json(), async (req, res) => {
  try {
    const { position, keywords } = req.body;
    console.log("Received position:", position);
    console.log("Received keywords:", keywords);

    if (!position || !keywords?.length) {
      return res.status(400).json({ error: "Position and keywords are required" });
    }

    const chain = initialQuestionsPrompt
      .pipe(mistralModel)
      .pipe(new StringOutputParser());

    const questions = await chain.invoke({
      position,
      keywords: keywords.join(", "),
    });

    const sessionId = `session_${Date.now()}`;

    sessions.set(sessionId, {
      position,
      keywords,
      questions: questions.split("\n").filter(q => q.trim()),
      currentQuestionIndex: 0,
      answers: [],
    });

    res.json({
      success: true,
      sessionId,
      firstQuestion: sessions.get(sessionId).questions[0],
    });
  } catch (error) {
    console.error("Interview Start Error:", error);
    res.status(500).json({ error: "Failed to start interview" });
  }
});

// Generate Questions
app.post('/api/generate-questions', express.json(), async (req, res) => {
  try {
    const { position, keywords, numQuestions = 5 } = req.body;

    if (!position || !keywords?.length) {
      return res.status(400).json({ 
        error: "Position and keywords are required" 
      });
    }

    const chain = interviewPrompt
      .pipe(mistralModel)
      .pipe(new StringOutputParser());

    const questions = await chain.invoke({
      position,
      keywords: keywords.join(", "),
      numQuestions,
    });

    res.json({
      success: true,
      questions: questions.split('\n').filter(q => q.trim()),
    });
  } catch (error) {
    console.error("Question Generation Error:", error);
    res.status(500).json({
      error: "Failed to generate questions",
      details: error.message,
    });
  }
});

// Evaluate Answer
app.post("/evaluate-answer", express.json(), async (req, res) => {
  try {
    const { sessionId, answer } = req.body;
    const session = sessions.get(sessionId);

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }

    const currentQuestion = session.questions[session.currentQuestionIndex];
    if (!currentQuestion || !answer) {
      return res.status(400).json({ error: "Missing question or answer" });
    }

    const formatInstructions = evaluationParser.getFormatInstructions();
    const prompt = await evaluationPrompt.format({
      question: currentQuestion,
      answer: answer,
      format_instructions: formatInstructions
    });

    const result = await mistralModel.invoke(prompt);
    const evaluationData = await evaluationParser.parse(result.content);

    session.answers.push({
      question: currentQuestion,
      answer,
      evaluation: evaluationData,
    });

    const shouldContinue = session.currentQuestionIndex < session.questions.length - 1;
    const response = {
      success: true,
      evaluation: evaluationData,
      isComplete: !shouldContinue
    };

    if (shouldContinue) {
      session.currentQuestionIndex++;
      response.nextQuestion = session.questions[session.currentQuestionIndex];
    } else {
      response.summary = session.answers;
    }

    res.json(response);
  } catch (error) {
    console.error("Evaluation Error:", error);
    // Fallback response
    res.json({
      success: true,
      evaluation: {
        score: 2,
        rating: "Fair",
        feedback: "The answer showed basic understanding but needs improvement",
        followUp: "Can you explain this concept in more detail?"
      },
      isComplete: false
    });
  }
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});