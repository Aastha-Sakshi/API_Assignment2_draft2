# Demo voice-over script

For `docs/demo_titled.mp4` — 6 min 35 s. Roughly 730 words, which is a relaxed
pace with pauses left in. Timecodes are where each card appears, so you have a
moment of dark screen to start each line on.

If you drift behind, the section cards are the place to catch up — nothing is
happening on screen during them.

> **One thing not to say:** don't claim the big model is slower. In this
> recording both answered in about a second, so the picture would contradict
> you. The honest contrast is cost and independence, not speed.

---

### 0:00 — Title card

Hi. This is my AI Recruitment Assistant, built for Assignment Two.

### 0:08 — What this project does

The idea is simple. A recruiter gets hundreds of CVs, and each one has to be
judged against a job description. This does that automatically.

### 0:13 — The API

Everything runs as a web service. Each thing the system can do is its own
endpoint, and they're grouped by the kind of AI doing the work.

The first two are vision tasks — they treat the document as an image. The rest
are language tasks — they read the text.

There are twelve endpoints in all. And one right at the bottom that runs the
entire chain in a single call.

### 0:47 — Which models, and why

Rather than just telling you which models I used, the service will tell you
itself. This lists every model, the job it does, and which category it belongs
to. So nothing in my report has to be taken on trust — you can check it against
the thing that's actually running.

### 1:03 — Is it healthy?

There's a health check too. It says whether the language model service is
reachable, and whether my own trained model has loaded. If something's broken,
you find out here, rather than halfway through a request.

### 1:21 — The app

This is the front end. One tab per task. Worth saying: this page contains no AI
at all. It only calls the API. All the intelligence lives behind it.

### 1:33 — Reading the resume

So let's start. I upload a CV as a PDF.

Two things happen. First it checks the file really is a CV, by looking at the
shape of the page — the headings, the columns, the whitespace.

Then it reads the text out. If the PDF has a text layer, it just takes it, and
that's instant. If it's a scan or a photo, it falls back to proper image
recognition. Here it found a text layer, so no scanning was needed.

And this is the text it pulled out.

### 2:16 — Pulling out the facts

Next it picks out the useful things — skills, job titles, dates, companies.

But notice what it does with the personal details. The name, the email, the
phone number, the location — those are found and deliberately held back.
They're counted, not used. In a minute you'll see why that matters.

### 2:42 — Scoring the fit

Now the interesting part. This is a model I trained myself, on about six
thousand real CV and job-description pairs.

It gives one of three answers — good fit, potential fit, or no fit — with a
score for each one. It runs locally, on an ordinary processor, in about a
second.

### 3:04 — Small model vs large model

And here's the experiment at the heart of the project.

I give the same CV to a much larger model — twenty billion parameters, against
my sixty-six million. Around three hundred times bigger.

And they disagree. Mine says no fit. The large one says good fit.

So which is right? Honestly — I can't tell you. I tested both on three hundred
examples, and the gap between them was small enough to be chance.

What I can say is that mine costs nothing and never leaves my machine.

### 3:39 — Asking questions

You can also just ask questions about the CV.

The answers are pulled straight out of the document — real quotes, not the
model's own words. So it can't invent a qualification the candidate never
claimed.

Now watch what happens when I ask something the CV doesn't answer — what salary
they're expecting.

It doesn't guess. It says it isn't in the document. That refusal is the whole
point.

### 4:20 — Writing the brief

This writes a short summary for the recruiter — strengths, gaps, and questions
worth asking in an interview.

And here's why those personal details mattered earlier. Look at the summary —
the candidate's name has been taken out before any of this text was sent to the
language model.

That model is hosted by someone else, so whatever I send leaves my machine. The
name isn't needed to judge the CV, so it doesn't go.

### 5:03 — All of it, in one call

Everything so far was one task at a time. This runs the whole chain in a single
request — checks the document, reads it, pulls out the facts, scores the fit,
writes the brief.

And that's really the point of the project. Six AI tasks that feed into each
other, rather than six separate demos sitting side by side. Take any one away
and the ones after it stop working.

### 5:47 — Watching it in production

Finally, the part that's easy to skip.

Anything running for real needs watching. So the service measures itself — how
long requests take, how many fail, how many tokens it used and what they cost,
and how confident it is. All of that from real traffic.

The last one is different. Quality can't be measured from live traffic, because
you'd need to already know the right answers. So that one runs offline, and
that's the result you can see here.

Thanks for watching.
