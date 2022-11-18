from enum import Enum


class step(Enum):
  out = 0
  over = 1
  into = 2


class graph_stack:
    def __init__(self, root):
      self.S = [root]

  # basic query
    def is_valid(self) -> bool:
      return len(self.S)>0
   
    def is_at_root_level(self) -> bool:
      return len(self.S)==1
  

  # stack functions
    def push_level(self, x):
      self.S.append(x)
 
    def pop_level(self):
      self.S.pop(-1)


  # accessors
    def current_level(self):
      assert self.is_valid()
      return self.S[-1]

    def parent_level(self):
      assert self.is_valid() and not self.is_at_root_level()
      return self.S[-2]


class graph_traversal_stack:
    def __init__(self, g, adaptor):
      self.g = g
      self.adaptor = adaptor
      self.S = graph_stack([adaptor.root_nodes(g), 0])


    #TODO change access to g
    def current_node(self):
      lvl_nodes,lvl_pos = self.S.current_level()
      return self.g[0][lvl_nodes[lvl_pos]]
    def parent_node(self):
      parent_nodes,parent_pos = self.S.parent_level()
      return self.g[0][parent_nodes[parent_pos]]

    def advance_current_node(self):
      self.S.current_level()[1] += 1
    def advance_current_node_to_last(self):
      self.S.current_level()[1] = len(self.S.current_level()[0])


    def push_level(self, children):
      self.S.push_level([children, 0])

    def pop_level(self):
      self.S.pop_level()

    def level_is_done(self) -> bool:
      lvl_nodes,lvl_pos = self.S.current_level()
      return len(lvl_nodes) == lvl_pos

    def is_at_root_level(self) -> bool:
      return self.S.is_at_root_level()

    def is_done(self) -> bool:
      return self.is_at_root_level() and self.level_is_done()


def unwind(S, f):
  while not S.is_at_root_level():
    v = S.current_node()
    parent = S.parent_node()
    f.post(v)
    f.up(v,parent)
    S.pop_level()

  v = S.current_node()
  f.post(v)


def depth_first_search(g, f, adaptor):
  S = graph_traversal_stack(g,adaptor)

  while not S.is_done():
    if not S.level_is_done():
      v = S.current_node()
      next_step = f.pre(v)
      if next_step == step.out: # stop
        matching_node = S.current_node()
        unwind(S,f)
        return matching_node
      if next_step == step.over:  # prune
        S.push_level(adaptor.children(v))
        S.advance_current_node_to_last()
      if next_step == step.into:  # go down
        S.push_level(adaptor.children(v))
        if not S.level_is_done():
          f.down(v,adaptor.first_child(g,v))

    else:
      S.pop_level()
      v = S.current_node()
      f.post(v)
      S.advance_current_node()
      if not S.is_at_root_level():
        parent = S.parent_node()
        f.up(v,parent)
        if not S.level_is_done():
          w = S.current_node()
          f.down(parent,w)


#// adaptation of general algorithm to find,prune and scan {
#
#/// The general algorithm asks if it should step out/over/into
#/// So the visitor's `pre` function must return a value of `step` type
#/// Here, we handle simpler `pre` functions:
#/// - find: return true to step out, return false to continue
#/// - prune: return true to step over, return false to continue
#/// - scan : always continue
#template<class F, auto convert_to_step>
#class depth_first_visitor_adaptor {
#  public:
#    constexpr
#    depth_first_visitor_adaptor(auto&& f)
#      : f(FWD(f))
#    {}
#
#    constexpr auto
#    pre(auto&& na) -> step {
#      return convert_to_step(f,na);
#    }
#    constexpr auto
#    down(auto&& na_above, auto&& na_below) -> void {
#      f.down(na_above,na_below);
#    }
#    constexpr auto
#    up(auto&& na_below, auto&& na_above) -> void {
#      f.up(na_below,na_above);
#    }
#    constexpr auto
#    post(auto&& na) -> void {
#      f.post(na);
#    }
#  private:
#    remove_rvalue_reference<F> f;
#};
#
#template<class Graph_iterator_stack, class F> constexpr auto
#// requires Graph_iterator_stack is Array<Iterator_range<Graph>>
#depth_first_find_adjacency_stack(Graph_iterator_stack& S, F&& f) {
#  constexpr auto convert_to_step = [](auto&& f, auto&& na){ return f.pre(na) ? step::out : step::into; };
#  depth_first_visitor_adaptor<F,convert_to_step> vis(FWD(f));
#  return depth_first_search(S,vis);
#}
#template<class Graph_iterator_stack, class F> auto
#// requires Graph_iterator_stack is Array<Iterator_range<Graph>>
#depth_first_prune_adjacency_stack(Graph_iterator_stack& S, F&& f) -> void {
#  constexpr auto convert_to_step = [](auto&& f, auto&& na){ return f.pre(na) ? step::over: step::into; };
#  depth_first_visitor_adaptor<F,convert_to_step> vis(FWD(f));
#  depth_first_search(S,vis);
#}
#template<class Graph_iterator_stack, class F> auto
#// requires Graph_iterator_stack is Array<Iterator_range<Graph>>
#depth_first_scan_adjacency_stack(Graph_iterator_stack& S, F && f) -> void {
#  constexpr auto convert_to_step = [](auto&& f, auto&& na){ f.pre(na); return step::into; };
#  depth_first_visitor_adaptor<F,convert_to_step> vis(FWD(f));
#  depth_first_search(S,vis);
#}
#// adaptation of general algorithm to find,prune and scan }
#
#
#// only pre or post scans {
#template<class F>
#class preorder_visitor_adaptor {
#  public:
#    template<class F_0> constexpr
#    preorder_visitor_adaptor(F_0&& f)
#      : f(FWD(f))
#    {}
#
#    template<class Node_adjacency> constexpr auto
#    pre(Node_adjacency&& na) {
#      return f(na);
#    }
#
#    template<class Node_adjacency> constexpr auto
#    down(Node_adjacency&&, Node_adjacency&&) {}
#    template<class Node_adjacency> constexpr auto
#    up(Node_adjacency&&, Node_adjacency&&) {}
#    template<class Node_adjacency> constexpr auto
#    post(Node_adjacency&&) {}
#  private:
#    remove_rvalue_reference<F> f;
#};
#
#template<class F>
#class postorder_visitor_adaptor {
#  public:
#    template<class F_0> constexpr
#    postorder_visitor_adaptor(F_0&& f)
#      : f(FWD(f))
#    {}
#
#    template<class Node_adjacency> constexpr auto
#    pre(Node_adjacency&&) {}
#    template<class Node_adjacency> constexpr auto
#    down(Node_adjacency&&, Node_adjacency&&) {}
#    template<class Node_adjacency> constexpr auto
#    up(Node_adjacency&&, Node_adjacency&&) {}
#
#    template<class Node_adjacency> constexpr auto
#    post(Node_adjacency&& na) {
#      return f(na);
#    }
#  private:
#    remove_rvalue_reference<F> f;
#};
#
#template<class Graph_iterator_stack, class F> constexpr auto
#preorder_depth_first_scan_adjacency_stack(Graph_iterator_stack& S, F&& f) -> void {
#  preorder_visitor_adaptor<F> pre_vis(FWD(f));
#  prepostorder_depth_first_scan_adjacency_stack(S,pre_vis);
#}
#template<class Graph_iterator_stack, class F> constexpr auto
#postorder_depth_first_scan_adjacency_stack(Graph_iterator_stack& S, F&& f) -> void {
#  postorder_visitor_adaptor<F> post_vis(FWD(f));
#  prepostorder_depth_first_scan_adjacency_stack(S,post_vis);
#}
#
#template<class Graph_iterator_stack, class F> constexpr auto
#preorder_depth_first_prune_adjacency_stack(Graph_iterator_stack& S, F&& f) -> void {
#  preorder_visitor_adaptor<F> pre_vis(FWD(f));
#  prepostorder_depth_first_prune_adjacency_stack(S,pre_vis);
#}
#// only pre or post scans }
#
#
#} // std_e
