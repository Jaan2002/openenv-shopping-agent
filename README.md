# Smart AI Shopping Assistant (OpenEnv Environment)

## 📌 Overview
This project implements a real-world OpenEnv environment where an AI agent learns to make optimal shopping decisions based on user needs, budget constraints, and product features.

The environment simulates realistic e-commerce scenarios and evaluates AI agents using structured tasks, reward functions, and deterministic graders.

---

## 🎯 Objective
To build a complete OpenEnv-compliant environment that allows AI agents to:
- Analyze user requirements
- Compare multiple products
- Make optimal decisions under constraints
- Provide reasoning for selections

---

## 🌍 Real-World Utility
This environment simulates real-world multi-criteria decision making in e-commerce, enabling evaluation of AI agents under constraints such as budget, product features, and user preferences.

The environment supports multi-category decision making (smartphones, laptops, accessories), enabling evaluation of AI agents across diverse shopping scenarios.

---

## 🧠 Environment Design

### OpenEnv APIs Implemented:
- `reset()` → Initializes a new task
- `step(action)` → Executes action and returns reward
- `state()` → Returns current environment state

---

## 📦 Observation Space

Each observation includes:
- `category` → Product category (e.g., smartphone, laptop)
- `user_need` → User requirement
- `budget` → Maximum budget
- `priority` → Decision factor (price, rating, battery)
- `products` → List of available products with attributes

---

## 🎮 Action Space

The agent selects:
- `action_type` → Product name
- `explanation` → Reasoning for the decision

---

## 🧪 Tasks (Difficulty Levels)

### 🟢 Easy
- Budget-based decision (smartphone under ₹10,000)

### 🟡 Medium
- Value-based decision (laptop under ₹50,000)

### 🔴 Hard
- Multi-constraint decision (headphones based on battery, price)

---

## 🏁 Reward Function

The reward system evaluates agent performance:

- ✅ Optimal choice → **1.0**
- ⚖️ Partial correct → **0.5**
- ❌ Incorrect choice → **Penalty**
- 💡 Explanation quality bonus → +0.2
- 💸 Budget violation penalty → -0.3

Rewards are normalized between **0.0 and 1.0**

---

## 🤖 AI Integration

The environment uses an LLM-based agent (via OpenRouter/OpenAI client) to:
- Analyze inputs
- Select optimal products
- Generate reasoning

Fallback logic ensures the system remains stable even if API calls fail.

---

## 📊 Evaluation

The environment evaluates agent performance across:
- Multiple tasks (easy → medium → hard)
- Deterministic grading logic
- Final cumulative score across tasks

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
