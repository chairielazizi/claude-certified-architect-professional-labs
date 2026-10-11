# Diagrams - Claims Correspondence Copilot v1.5.0

Generated from `architecture_spec.json`; paste into any Mermaid renderer (GitHub renders these natively).

## C4 level 1 - system context

```mermaid
flowchart LR
    adjuster(["Claims adjuster"])
    team_lead(["Team lead"])
    ccc["Claims Correspondence Copilot<br/>v1.5.0"]
    claimcore["ClaimCore<br/>Claims system of record (REST API)"]
    docvault["DocVault<br/>Policy wording and claim documents (S3)"]
    postmaster["PostMaster<br/>Outbound correspondence service - NOT called by the copilot (by design)"]
    okta["Okta<br/>Workforce SSO"]
    anthropic["Claude API<br/>Anthropic Messages API (US region)"]
    adjuster -->|uses| ccc
    team_lead -->|uses| ccc
    ccc --> claimcore
    ccc --> docvault
    ccc --> okta
    ccc --> anthropic
    ccc -.->|not called - by design| postmaster
    classDef ext fill:#eeeeee,stroke:#999999
    class claimcore,docvault,postmaster,okta,anthropic ext
```

## C4 level 2 - containers

```mermaid
flowchart LR
    subgraph people["People"]
        adjuster(["Claims adjuster"])
        team_lead(["Team lead"])
    end
    subgraph system["Claims Correspondence Copilot"]
        workspace_ui["Adjuster workspace plug-in<br/>React micro-frontend"]
        drafting_api["Drafting API<br/>Python, FastAPI"]
        context_builder["Context builder<br/>Python"]
        drafting_agent{{"Drafting orchestrator<br/>Claude API (Sonnet 5.5) with read-only tools"}}
        guardrails["Guardrail service<br/>Python rules + Claude Haiku 5.5 classifier"]
        eval_runner[/"Eval and judge runner<br/>Python, Message Batches API"/]
        batch_backlog[/"Backlog batch drafter<br/>Python, Message Batches API"/]
    end
    subgraph stores["Data stores"]
        audit_store[("Audit store<br/>PostgreSQL (append-only)")]
        prompt_registry[("Prompt registry<br/>Git repository with signed tags")]
        eval_store[("Eval results<br/>S3 bucket")]
    end
    subgraph external["External systems"]
        claimcore["ClaimCore"]
        docvault["DocVault"]
        okta["Okta"]
        anthropic["Claude API"]
    end
    adjuster --> workspace_ui
    team_lead --> workspace_ui
    workspace_ui --> drafting_api
    workspace_ui --> okta
    drafting_api --> context_builder
    drafting_api --> drafting_agent
    drafting_api --> guardrails
    drafting_api --> audit_store
    drafting_api --> claimcore
    context_builder --> claimcore
    context_builder --> docvault
    drafting_agent --> anthropic
    drafting_agent --> prompt_registry
    guardrails --> anthropic
    guardrails --> prompt_registry
    eval_runner --> anthropic
    eval_runner --> audit_store
    eval_runner --> eval_store
    batch_backlog --> context_builder
    batch_backlog --> guardrails
    batch_backlog --> audit_store
    batch_backlog --> anthropic
```

## Sequence - one drafting request through the agent

```mermaid
sequenceDiagram
    actor adjuster as Claims adjuster
    participant workspace_ui as Adjuster workspace plug-in
    participant drafting_api as Drafting API
    participant context_builder as Context builder
    participant claimcore as ClaimCore
    participant docvault as DocVault
    participant drafting_agent as Drafting orchestrator
    participant anthropic as Claude API
    participant guardrails as Guardrail service
    participant audit_store as Audit store
    adjuster->>workspace_ui: Draft letter (claim id, letter type)
    workspace_ui->>drafting_api: POST /drafts with SSO token
    drafting_api->>context_builder: Build claim-scoped context
    context_builder->>claimcore: GET claim and latest notes (OAuth2)
    context_builder->>docvault: GET policy wording (product, version)
    drafting_api->>drafting_agent: Draft(context, prompt version)
    drafting_agent->>anthropic: messages.stream - Sonnet 5.5, cached policy prefix
    anthropic-->>drafting_agent: Streamed draft tokens
    drafting_api->>guardrails: Check draft (rules + Haiku classifier)
    drafting_api->>drafting_api: Field validator vs ClaimCore values
    drafting_api->>audit_store: Append draft, model and prompt versions
    drafting_api->>claimcore: save_draft as pending correspondence
    drafting_api-->>workspace_ui: Stream approved-for-review draft
    adjuster->>workspace_ui: Edit and approve (sent via PostMaster workflow)
```

_Components in the container view: 7; request-flow steps: 14._
