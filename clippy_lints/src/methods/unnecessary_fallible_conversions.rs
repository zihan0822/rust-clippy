use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, snippet};
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_parent_expr, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{GenericArgsRef, Ty};

use rustc_span::Span;

use super::UNNECESSARY_FALLIBLE_CONVERSIONS;

#[derive(Copy, Clone)]
enum SpansKind {
    TraitFn { trait_span: Span, fn_span: Span },
    Fn { fn_span: Span },
}

/// What function is being called and whether that call is written as a method call or a function
/// call
#[derive(Copy, Clone)]
#[expect(clippy::enum_variant_names)]
enum FunctionKind<'tcx> {
    /// `T::try_from(U)`
    TryFromFunction(Option<SpansKind>),
    /// `t.try_into()`
    TryIntoMethod,
    /// `U::try_into(t)`
    TryIntoFunction(Option<SpansKind>),
    /// `T::from_str(s)`
    FromStrFunction(Option<SpansKind>),
    /// `s.parse::<U>()`
    /// Contains (receiver, target type)
    StrParseMethod(&'tcx Expr<'tcx>, Ty<'tcx>),
}

impl FunctionKind<'_> {
    fn appl_sugg(
        &self,
        cx: &LateContext<'_>,
        parent_unwrap_call: Option<Span>,
        primary_span: Span,
    ) -> (Applicability, Vec<(Span, String)>) {
        let Some(unwrap_span) = parent_unwrap_call else {
            return (Applicability::Unspecified, self.default_sugg(primary_span));
        };

        match &self {
            FunctionKind::TryFromFunction(None)
            | FunctionKind::TryIntoFunction(None)
            | FunctionKind::FromStrFunction(None) => (Applicability::Unspecified, self.default_sugg(primary_span)),
            _ => (
                Applicability::MachineApplicable,
                self.machine_applicable_sugg(cx, primary_span, unwrap_span),
            ),
        }
    }

    fn default_sugg(&self, primary_span: Span) -> Vec<(Span, String)> {
        let replacement = match *self {
            FunctionKind::TryFromFunction(_) | FunctionKind::FromStrFunction(_) | FunctionKind::StrParseMethod(..) => {
                "From::from"
            },
            FunctionKind::TryIntoFunction(_) => "Into::into",
            FunctionKind::TryIntoMethod => "into",
        };

        vec![(primary_span, String::from(replacement))]
    }

    fn machine_applicable_sugg(
        &self,
        cx: &LateContext<'_>,
        primary_span: Span,
        unwrap_span: Span,
    ) -> Vec<(Span, String)> {
        let (trait_name, fn_name) = match self {
            FunctionKind::TryFromFunction(_) | FunctionKind::FromStrFunction(_) | FunctionKind::StrParseMethod(..) => {
                ("From".to_owned(), "from".to_owned())
            },
            FunctionKind::TryIntoFunction(_) | FunctionKind::TryIntoMethod => ("Into".to_owned(), "into".to_owned()),
        };

        let mut sugg = match *self {
            FunctionKind::TryFromFunction(Some(spans))
            | FunctionKind::TryIntoFunction(Some(spans))
            | FunctionKind::FromStrFunction(Some(spans)) => match spans {
                SpansKind::TraitFn { trait_span, fn_span } => vec![(trait_span, trait_name), (fn_span, fn_name)],
                SpansKind::Fn { fn_span } => vec![(fn_span, fn_name)],
            },
            FunctionKind::TryIntoMethod => vec![(primary_span, fn_name)],
            FunctionKind::StrParseMethod(receiver, target_ty) => {
                let adjustments = cx.typeck_results().expr_adjustments(receiver);
                let deref_count = adjustments
                    .iter()
                    .take_while(|adj| matches!(adj.kind, Adjust::Deref(_)))
                    .count();
                let prefix = if deref_count == 0 {
                    ""
                } else {
                    &format!("&{:*>deref_count$}", "")
                };
                let receiver_snippet = receiver.span.get_source_text(cx).unwrap();
                let from_call = format!("{}::from({prefix}{receiver_snippet})", target_ty.to_string());
                vec![(primary_span, from_call)]
            },
            // Or the suggestion is not machine-applicable
            _ => unreachable!(),
        };

        sugg.push((unwrap_span, String::new()));
        sugg
    }

    fn mk_args<'tcx>(&self, cx: &LateContext<'tcx>, node_args: GenericArgsRef<'tcx>) -> GenericArgsRef<'tcx> {
        match self {
            FunctionKind::FromStrFunction(_) | FunctionKind::StrParseMethod(..) => {
                let str_ref_ty = Ty::new_imm_ref(cx.tcx, cx.tcx.lifetimes.re_erased, cx.tcx.types.str_);
                cx.tcx
                    .mk_args_from_iter(node_args.iter().chain(std::iter::once(str_ref_ty.into())))
            },
            _ => node_args,
        }
    }
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
    node_args: GenericArgsRef<'tcx>,
    kind: FunctionKind<'_>,
    primary_span: Span,
) {
    if let &[self_ty, other_ty] = node_args.as_slice()
        // useless_conversion already warns `T::try_from(T)`, so ignore it here
        && self_ty != other_ty
        && let Some(self_ty) = self_ty.as_type()
        && let Some(from_into_trait) = cx.tcx.get_diagnostic_item(match kind {
            FunctionKind::TryFromFunction(_) | FunctionKind::FromStrFunction(_) |
                FunctionKind::StrParseMethod(..) => sym::From,
            FunctionKind::TryIntoMethod | FunctionKind::TryIntoFunction(_) => sym::Into,
        })
        // If `T: TryFrom<U>` and `T: From<U>` both exist, then that means that the `TryFrom`
        // _must_ be from the blanket impl and cannot have been manually implemented
        // (else there would be conflicting impls, even with #![feature(spec)]), so we don't even need to check
        // what `<T as TryFrom<U>>::Error` is: it's always `Infallible`
        && implements_trait(cx, self_ty, from_into_trait, &[other_ty])
        && let Some(other_ty) = other_ty.as_type()
    {
        // Extend the span to include the unwrap/expect call:
        // `foo.try_into().expect("..")`
        //      ^^^^^^^^^^^^^^^^^^^^^^^
        //
        // `try_into().unwrap()` specifically can be trivially replaced with just `into()`,
        // so that can be machine-applicable
        let parent_unwrap_call = get_parent_expr(cx, expr).and_then(|parent| {
            if let ExprKind::MethodCall(path, .., span) = parent.kind
                && let sym::unwrap | sym::expect = path.ident.name
            {
                // include `.` before `unwrap`/`expect`
                Some(span.with_lo(expr.span.hi()))
            } else {
                None
            }
        });

        // If there is an unwrap/expect call, extend the span to include the call
        let span = if let Some(unwrap_call) = parent_unwrap_call {
            primary_span.with_hi(unwrap_call.hi())
        } else {
            primary_span
        };

        let (source_ty, target_ty) = match kind {
            FunctionKind::TryIntoMethod | FunctionKind::TryIntoFunction(_) => (self_ty, other_ty),
            FunctionKind::TryFromFunction(_) | FunctionKind::FromStrFunction(_) | FunctionKind::StrParseMethod(..) => {
                (other_ty, self_ty)
            },
        };

        let (applicability, sugg) = kind.appl_sugg(cx, parent_unwrap_call, primary_span);

        span_lint_and_then(
            cx,
            UNNECESSARY_FALLIBLE_CONVERSIONS,
            span,
            "use of a fallible conversion when an infallible one could be used",
            |diag| {
                with_forced_trimmed_paths!({
                    diag.note(format!("converting `{source_ty}` to `{target_ty}` cannot fail"));
                });
                diag.multipart_suggestion("use", sugg, applicability);
            },
        );
    }
}

/// Checks method call exprs:
/// - `0i32.try_into()`
/// - `s.parse()`
pub(super) fn check_method<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if let ExprKind::MethodCall(path, receiver, ..) = expr.kind {
        let node_args = cx.typeck_results().node_args(expr.hir_id);
        let (kind, span) =
            if path.ident.name == sym::parse && cx.typeck_results().expr_ty_adjusted(receiver).peel_refs().is_str() {
                let Some(target_ty) = node_args.first().and_then(|arg| arg.as_type()) else {
                    return;
                };
                (FunctionKind::StrParseMethod(receiver, target_ty), expr.span)
            } else {
                (FunctionKind::TryIntoMethod, path.ident.span)
            };
        check(cx, expr, kind.mk_args(cx, node_args), kind, span);
    }
}

/// Checks function call exprs:
/// - `<i64 as TryFrom<_>>::try_from(0i32)`
/// - `<_ as TryInto<i64>>::try_into(0i32)`
pub(super) fn check_function(cx: &LateContext<'_>, expr: &Expr<'_>, callee: &Expr<'_>) {
    if let ExprKind::Path(ref qpath) = callee.kind
        && let Some(item_def_id) = cx.qpath_res(qpath, callee.hir_id).opt_def_id()
        && let Some(trait_def_id) = cx.tcx.trait_of_assoc(item_def_id)
    {
        let qpath_spans = match qpath {
            QPath::Resolved(_, path) => {
                if let [trait_seg, fn_seg] = path.segments {
                    Some(SpansKind::TraitFn {
                        trait_span: trait_seg.ident.span,
                        fn_span: fn_seg.ident.span,
                    })
                } else {
                    None
                }
            },
            QPath::TypeRelative(_, seg) => Some(SpansKind::Fn {
                fn_span: seg.ident.span,
            }),
            QPath::LangItem(_, _) => unreachable!("`TryFrom` and `TryInto` are not lang items"),
        };
        let func_kind = if cx.tcx.is_diagnostic_item(sym::from_str_method, item_def_id) {
            FunctionKind::FromStrFunction(qpath_spans)
        } else {
            match cx.tcx.get_diagnostic_name(trait_def_id) {
                Some(sym::TryFrom) => FunctionKind::TryFromFunction(qpath_spans),
                Some(sym::TryInto) => FunctionKind::TryIntoFunction(qpath_spans),
                _ => return,
            }
        };
        let node_args = func_kind.mk_args(cx, cx.typeck_results().node_args(callee.hir_id));

        check(cx, expr, node_args, func_kind, callee.span);
    }
}
