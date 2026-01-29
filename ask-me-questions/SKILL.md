---
name: ask-me-questions
description: |
  Interactive questioning skill to help users understand vibecoded or AI-generated codebases they've lost track of. Use when:
  (1) User is confused about how something works in their codebase
  (2) User doesn't understand code that was generated with too much AI autonomy
  (3) User wants to find gaps, errors, or misunderstandings in implementation
  (4) User says things like "I don't understand this", "how does X work", "what does this do"
  (5) User wants to audit their understanding of AI-generated code
  (6) User mentions "vibecoding", "lost track", "don't know what the agent built"
  Triggers: "confused", "don't understand", "how does this work", "explain", "audit my understanding", "vibecoded", "what did the agent do"
---

# Ask Me Questions

Help users regain understanding of codebases where AI autonomy outpaced human oversight.

## Core Workflow

### Step 1: Understand the Confusion

When user invokes this skill, first identify:
- What specific area/feature/file they're confused about
- How deep their confusion goes (surface-level vs fundamental)
- Whether they built it themselves, had AI build it, or inherited it

**Ask immediately:**
- "What part of the codebase are you confused about?"
- "Is this code you wrote, AI-generated, or inherited?"

### Step 2: Explore the Relevant Code

Before asking detailed questions:
1. Read the files/modules the user is confused about
2. Trace the data flow and dependencies
3. Identify key abstractions and patterns used
4. Note anything that looks suspicious, inconsistent, or overly complex

### Step 3: Ask Clarifying Questions

Use the environment's question-asking tool (AskUserQuestion in Claude Code, or equivalent).

**Question categories to cover:**

#### Intent Questions
- "What did you want this feature to accomplish?"
- "What problem was this supposed to solve?"
- "Who is the intended user of this functionality?"

#### Behavior Questions
- "What should happen when [specific action]?"
- "What's the expected output when [input scenario]?"
- "How should errors be handled here?"

#### Architecture Questions
- "Why is [component A] separate from [component B]?"
- "Should this data be stored in [location] or elsewhere?"
- "Is [dependency X] actually needed here?"

#### Verification Questions
- "Is it correct that [observed behavior]?"
- "Did you intend for [specific implementation detail]?"
- "Should [edge case] be handled differently?"

### Step 4: Surface Gaps and Issues

Based on code exploration and user answers, identify:

1. **Understanding gaps** - Things the user doesn't know about their own code
2. **Implementation gaps** - Missing error handling, edge cases, validation
3. **Consistency issues** - Patterns that don't match, naming inconsistencies
4. **Dead code** - Unused functions, unreachable branches
5. **Security concerns** - Exposed secrets, missing auth, injection risks
6. **Over-engineering** - Unnecessary abstractions, premature optimization

**Report findings clearly:**
```
Based on our conversation, I found:

Understanding gaps:
- You weren't aware that X calls Y, which means Z

Implementation issues:
- No error handling in /src/api/users.ts:45
- Edge case: empty array not handled in processItems()

Potential concerns:
- API key appears hardcoded in config.ts
```

### Step 5: Resolve and Educate

For each gap/issue:
1. Explain what the code actually does (vs what user thought)
2. Ask if the current behavior is intentional
3. Offer to fix issues or explain the fix
4. Help user understand the pattern for future reference

## Question-Asking Guidelines

### Use the Right Tool
- **Claude Code**: Use `AskUserQuestion` tool with clear options
- **Cursor**: Use the built-in question mechanism
- **Other environments**: Ask inline and wait for response

### Question Quality
- **One topic at a time** - Don't overwhelm with multiple complex questions
- **Concrete, not abstract** - "What should /api/users return?" not "How should the API work?"
- **Include context** - Reference specific files, lines, or behaviors
- **Offer options when possible** - Multiple choice is faster than open-ended

### Example Question Flow

```
Q1: "What part of the codebase confuses you most?"
→ User: "The authentication flow"

Q2: "I see auth is handled in /src/lib/auth.ts and /src/middleware/session.ts.
    Which aspect is unclear?"
   - How users log in
   - How sessions are managed
   - How protected routes work
   - All of it
→ User: "How sessions are managed"

Q3: "Sessions are stored in Redis with a 24h TTL. Is that intentional,
    or did you expect something different?"
→ User: "I didn't know it used Redis, I thought it was cookies"

[Found understanding gap - explain and verify]
```

## Common Vibecoding Issues to Check

When auditing AI-generated code, watch for:

- **Hallucinated APIs** - Calls to functions/endpoints that don't exist
- **Incomplete implementations** - TODO comments, placeholder code
- **Inconsistent patterns** - Multiple ways of doing the same thing
- **Missing validation** - User input not sanitized
- **Hardcoded values** - Magic numbers, embedded credentials
- **Overly clever code** - Complex solutions to simple problems
- **Copy-paste artifacts** - Duplicated code with slight variations
- **Broken error handling** - Catches that swallow errors silently
- **Type mismatches** - Incorrect TypeScript/type annotations
- **Stale dependencies** - Outdated or unused packages
