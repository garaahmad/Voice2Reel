# Current cartoon presenter

The user rejected both the earliest minimalist study and the first fast presenter. Current work is **presenter v2**, documented in [PRESENTER_V2.md](PRESENTER_V2.md), with original red-beret artwork, a fuller face and consistent rounded limbs, separate shoulders, hands/feet, and cropped supersampled drawing.

Six supported gestures remain `present`, `point`, `think`, `shrug`, `thumbs_up`, `wave`. Face expressions, eye direction, blinks, pose transitions, brief recovery and weight shifts create presenter motion. This is a bounded 2D animation engine; it cannot invent arbitrary actions, interactions or multiple-character scenes.

Camera framing changes the actor layer while keeping text readable. Panels support code, lists, flow and comparison. Explicit `visual.focus` plus `visual.status` emphasize an item without inferring code correctness. A brief graphic wipe and timed sound effect accompany cuts. Cairo is bundled for Arabic/Latin/punctuation coverage, with its complete SIL OFL license.

Narration audio drives mouth opening; music and effects do not. There is no phoneme lip-sync. Online Edge speech supplies word boundaries for subtitle phrases; Piper/XTTS keep estimated phrase timing. The default online voice, offline alternatives and mix controls are explained in the guide.

The reviewed Arabic example is manual editorial work. Actual AI attempts are preserved separately and remain evidence of writing limitations. This design has not yet been approved by the user.
