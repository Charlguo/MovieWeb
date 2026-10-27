Copy `.env.example` to `.env` and set `GROQ_API_KEY` and `SERVICE_API_KEY`.

Start the vector database and the agent service:

```
docker compose up --build
```

The site reads both values from `.env`. Do not put the Groq key in source code.

### **Showcase:**

![](<assets/Screenshot (443)+.png>)
![](<assets/Screenshot (442)+.png>)
![](<assets/Screenshot (447)+.png>)
![](<assets/Screenshot (448)+.png>)
![](<assets/Screenshot (445)+.png>)
![](<assets/Screenshot (446)+.png>)
