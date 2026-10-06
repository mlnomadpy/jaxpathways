# Learn JAX with your agent

These skills teach from a **local JAX Pathways checkout**. Installing them does not download the course, install Python packages or provide a hosted execution service.

You can also use the **portable course workspace** from the landing page (`public/downloads/jax-course-workspace.zip`). Extract it, open its `jaxpathways` folder in your agent and run the local installation command below from that folder. It contains the canonical lessons, projects and skills without Git history. The tutor uses it as the course root.

| Skill | Use it for |
| --- | --- |
| `start-learning` | Choose a route, identify your starting point, or resume your learner-owned `LEARNING.md` |
| `learn-jax` | Work through an authored lesson: prediction, mechanism, modification and evidence |
| `jax-course-guide` | Find the smallest useful lesson sequence for a topic, error or career goal |
| `check-jax` | Check understanding through a formative quiz and learner-written practical change |
| `start-jax` | Compatibility name for earlier onboarding installations |
| `create-jax-course` | Author and verify course material; this is a contributor workflow |

## Install from your checkout

Read the desired `SKILL.md` files first. With Node.js/npm available, run this from the repository root; choose your agent host when prompted:

```sh
npx skills add . --skill start-learning learn-jax jax-course-guide check-jax
```

The [Skills CLI](https://github.com/vercel-labs/skills) supports local directories, GitHub repositories and host selection. This command uses the project installation scope by default. To select Codex explicitly, add `--agent codex`; global installation is an optional user choice, not required by the course.

For an accessible GitHub revision that contains these skills:

```sh
npx skills add mlnomadpy/jaxpathways --skill start-learning learn-jax jax-course-guide check-jax
```

A private repository requires access through your configured Git/GitHub credentials. The remote command is documented, not verified against unpublished local changes. Use the local command when those changes are only in your checkout. Do not paste tokens into a lesson or learner plan.

Without Node, extract `public/downloads/jax-tutor-skills.zip` and copy the selected folders into the skill location supported by your agent host. For Codex, that is normally `$CODEX_HOME/skills` or `~/.codex/skills`. Installation is optional; the browser lessons and downloadable Python/notebooks work independently.

## Start and resume

Keep the course checkout open in the agent workspace. In Codex choose `start-learning` or type `$start-learning`; in Claude Code use `/start-learning`. For other hosts, say “Use start-learning to help me learn JAX from this checkout.” The tutor selects available material and writes `LEARNING.md` in your chosen workspace. Next time, use `learn-jax` to continue that plan.

The tutor separates reading, quiz correctness, reference execution, learner-written work and review. Planned topics remain visible as gaps. CPU and logical-device execution do not validate TPU performance or award credentials. Browser-local progress and `LEARNING.md` are separate stores; importing evidence requires an explicit reconciliation.
